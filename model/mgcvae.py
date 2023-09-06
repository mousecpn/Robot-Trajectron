import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.model_utils import ModeKeys,rgetattr,rsetattr,CustomLR,exp_anneal,sigmoid_anneal,unpack_RNN_state,run_lstm_on_variable_length_seqs,mutual_inf_mc
from model.dynamics import SingleIntegrator
from model.discrete_latent import DiscreteLatent
from model.gmm3d import GMM3D

class MultimodalGenerativeCVAE(nn.Module):
    def __init__(self,
                 hyperparams,
                 device,
                 log_writer=None):
        super(MultimodalGenerativeCVAE,self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.node_modules = nn.ModuleDict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state']
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state.values()]))

        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))
        self.create_graphical_model()

        dyn_limits = hyperparams['dynamic']['limits']
        self.dynamic = SingleIntegrator(1./self.hyperparams['frequency'], dyn_limits, device, self.x_size)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model):
        self.node_modules[name] = model.to(self.device)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule('/node_history_encoder',
                           model=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule('/node_future_encoder',
                           model=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule('/node_future_encoder/initial_h',
                           model=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule('/node_future_encoder/initial_c',
                           model=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))


        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule('/p_z_x',
                               model=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule('/hx_to_z',
                           model=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule('/q_z_xy',
                               #                                           Node Future Encoder
                               model=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule('/hxy_to_z',
                           model=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule('/decoder/state_action',
                           model=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule( '/decoder/rnn_cell',
                           model=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule('/decoder/initial_h',
                           model=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule('/decoder/proj_to_GMM_log_pis',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule('/decoder/proj_to_GMM_mus',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule('/decoder/proj_to_GMM_log_sigmas',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule('/decoder/proj_to_GMM_corrs',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']*3))

        self.x_size = x_size
        self.z_size = z_size


    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    
    def create_graphical_model(self):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  })

        self.create_new_scheduler(name='latent.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

        self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar('%s/%s' % (str(), annealed_var.replace('.', '/')),
                                               rgetattr(self, annealed_var), self.curr_iter)

    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               labels,
                               labels_st,
                               first_history_indices) -> torch.Tensor:
        """
        Encodes input and output tensors for node.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: tuple(x, y_e, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - y_e: Encoded label / future of the node.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        x, y_e, y = None, None, None
        initial_dynamics = dict()

        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:3]
        node_vel = inputs[:, -1, 3:6]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        node_pos_st = inputs_st[:, -1, 0:3]
        node_vel_st = inputs_st[:, -1, 3:6]

        n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel

        self.dynamic.set_initial_condition(initial_dynamics)

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)

        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = labels_st


        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)

        return x, y_e, y, n_s_t0

    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules['/node_history_encoder'],
                                                      original_seqs=node_hist,
                                                      lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules['/node_future_encoder/initial_h']
        initial_c_model = self.node_modules['/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state


    def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules['/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules['/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules['/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules['/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(self, tensor) -> torch.Tensor:
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules['/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules['/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules['/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules['/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(self, mode, x, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM3D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules['/decoder/rnn_cell']
        initial_h_model = self.node_modules['/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules['/decoder/state_action'](n_s_t0)

        state = initial_state

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
        outputs = []

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM3D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t[...,0].reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 3
                ).permute(0, 2, 1, 3).reshape(-1, 3 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 3
                ).permute(0, 2, 1, 3).reshape(-1, 3 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, 3* num_components))


            # dec_inputs = [zx, mu_t]
            dec_inputs = [zx, a_t]
            outputs.append(a_t)
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        outputs = torch.stack(outputs,dim=1)

        a_dist = GMM3D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components, 3]))

        if self.hyperparams['dynamic']['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()            
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist, outputs
    
    def p_y_xz2z(self, mode, x, n_s_t0, z_stacked, z_T,
               num_samples, ph_limit=100, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param z_T: stop predicting at z_T
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM3D. If mode is Predict, also samples from the GMM.
        """
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules['/decoder/rnn_cell']
        initial_h_model = self.node_modules['/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules['/decoder/state_action'](n_s_t0)

        state = initial_state

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
        outputs = []

        T = 1

        ph = 0
        while True:
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM3D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t[...,0].reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 3
                ).permute(0, 2, 1, 3).reshape(-1, 3 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 3
                ).permute(0, 2, 1, 3).reshape(-1, 3 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, 3* num_components))


            # dec_inputs = [zx, mu_t]
            dec_inputs = [zx, a_t]
            outputs.append(a_t)
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
            ph += 1
            if ph > ph_limit:
                print("out ot ph_limit")
                break
            if ph == 1:
                pos_mus = self.dynamic.initial_conditions['pos'].unsqueeze(1)[:, None].repeat(num_samples, 1, num_components, 1)
            else:
                pos_mus += mus[-1].reshape(num_samples, -1, num_components, pred_dim) * self.dynamic.dt
            if pos_mus[...,-1].mean() < z_T:
                break

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)
        outputs = torch.stack(outputs,dim=1)

        a_dist = GMM3D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components, 3]))

        if self.hyperparams['dynamic']['distribution']:
            y_dist = self.dynamic.integrate_distribution2zT(a_dist, z_T)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()            
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist, outputs

    def encoder(self, mode, x, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, x, y_e)
        self.latent.p_dist = self.p_z_x(mode, x)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p(self.log_writer, '%s' % str(), self.curr_iter)
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(), 'kl'), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(self, mode, x, y, n_s_t0, z, labels, prediction_horizon, num_samples):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        y_dist, outputs = self.p_y_xz(mode, x, n_s_t0, z,
                             prediction_horizon, num_samples, num_components=num_components)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        mseloss,_ = torch.min(torch.mean((outputs.reshape((-1,)+labels.shape) -labels.unsqueeze(0))**2,dim=(2,3)),dim=0)
        mseloss = mseloss.mean()
        # prob = (y_dist.log_pis/torch.sum(y_dist.log_pis,-1,keepdim=True)).unsqueeze(-1)
        # pred = torch.sum(prob*y_dist.mus,dim=-2).squeeze(0)
        # mse = torch.sqrt(torch.mean((labels - pred)**2,dim=-2)).mean(0).sum(0)
        
        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(), 'log_p_yt_xz'), log_p_yt_xz, self.curr_iter)

        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        return log_p_y_xz, mseloss

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   prediction_horizon) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=labels,
                                                        labels_st=labels_st,
                                                        first_history_indices=first_history_indices)

        z, kl = self.encoder(mode, x, y_e)
        log_p_y_xz, mseloss = self.decoder(mode, x, y, n_s_t0, z,
                                  labels,  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'])

        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO + mseloss

        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(), 'log_p_y_xz'),
                                          log_p_y_xz_mean,
                                          self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(), 'mutual_information_q'),
                                       mutual_inf_q,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(), 'mutual_information_p'),
                                       mutual_inf_p,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(), 'log_likelihood'),
                                       log_likelihood,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(), 'loss'),
                                       loss,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(), self.curr_iter)
        return loss

    def eval_loss(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  prediction_horizon) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                            inputs=inputs,
                                                            inputs_st=inputs_st,
                                                            labels=labels,
                                                            labels_st=labels_st,
                                                            first_history_indices=first_history_indices)

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood

        return nll

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=None,
                                                        labels_st=None,
                                                        first_history_indices=first_history_indices)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, our_sampled_future = self.p_y_xz(mode, x, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)
        if dist == True:
            return y_dist, our_sampled_future
        return our_sampled_future

    def predict2(self,
                inputs,
                inputs_st,
                first_history_indices,
                z_T,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                ph_limit=100):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=None,
                                                        labels_st=None,
                                                        first_history_indices=first_history_indices)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, our_sampled_future = self.p_y_xz2z(mode, x, n_s_t0, z,
                                            z_T,
                                            num_samples,
                                            ph_limit,
                                            num_components,
                                            gmm_mode)
        if dist == True:
            return y_dist, our_sampled_future
        return our_sampled_future
    
    def get_latent(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   prediction_horizon) -> torch.Tensor:

        mode = ModeKeys.TRAIN

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=labels,
                                                        labels_st=labels_st,
                                                        first_history_indices=first_history_indices)
        return x
