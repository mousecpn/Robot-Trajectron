import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE

class Trajectron(object):
    def __init__(self, hyperparams, log_writer,
                 device, model=None):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.node_models_dict = dict()
        self.nodes = set()

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type]])
            )
        self.pred_state = self.hyperparams['pred_state']

        if model is None:
            self.model = MultimodalGenerativeCVAE(self.hyperparams,
                                            self.device,
                                            log_writer=self.log_writer)
        else:
            self.model = model

    # def set_environment(self, env):
    #     self.env = env

    #     self.node_models_dict.clear()
    #     edge_types = env.get_edge_types()

    #     for node_type in env.NodeType:
    #         # Only add a Model for NodeTypes we want to predict
    #         if node_type in self.pred_state.keys():
    #             self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
    #                                                                         node_type,
    #                                                                         self.hyperparams,
    #                                                                         self.device,
    #                                                                         log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        # for node_str, model in self.node_models_dict.items():
        self.model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        # for node_str, model in self.node_models_dict.items():
        self.model.set_annealing_params()

    def step_annealers(self):
        # if node_type is None:
        #     for node_type in self.node_models_dict:
        #         self.node_models_dict[node_type].step_annealers()
        # else:
        #     self.node_models_dict[node_type].step_annealers()
        self.model.step_annealers()

    def train_loss(self, batch, ph=None):
        if ph is None:
            ph = self.ph
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)

        # Run forward pass
        # model = self.node_models_dict[node_type]
        loss = self.model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                prediction_horizon=ph)

        return loss

    def eval_loss(self, batch):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)

        # Run forward pass
        # model = self.node_models_dict[node_type]
        nll = self.model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict(self,
                batch,
                ph,
                num_samples=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False):

        # There are no nodes of type present for timestep
        (first_history_index,
            x_t, y_t, x_st_t, y_st_t) = batch

        x = x_t.to(self.device)
        x_st_t = x_st_t.to(self.device)

        # Run forward pass
        predictions = self.model.predict(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    prediction_horizon=ph,
                                    num_samples=num_samples,
                                    z_mode=z_mode,
                                    gmm_mode=gmm_mode,
                                    full_dist=full_dist,
                                    all_z_sep=all_z_sep,
                                    dist=dist)
        if dist == True:
            y_dist, predictions = predictions

        predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            # for i, ts in enumerate(timesteps_o):
            #     if ts not in predictions_dict.keys():
            #         predictions_dict[ts] = dict()
            #     predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
        if dist:
            return y_dist, predictions_np
        return predictions_np

    def predict2(self,
                batch,
                z_T,
                num_samples=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                ph_limit=100):

        # There are no nodes of type present for timestep
        (first_history_index,
            x_t, y_t, x_st_t, y_st_t) = batch

        x = x_t.to(self.device)
        x_st_t = x_st_t.to(self.device)

        # Run forward pass
        predictions = self.model.predict2(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    z_T=z_T,
                                    num_samples=num_samples,
                                    z_mode=z_mode,
                                    gmm_mode=gmm_mode,
                                    full_dist=full_dist,
                                    all_z_sep=all_z_sep,
                                    dist=dist,
                                    ph_limit=100)
        if dist == True:
            y_dist, predictions = predictions

        predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            # for i, ts in enumerate(timesteps_o):
            #     if ts not in predictions_dict.keys():
            #         predictions_dict[ts] = dict()
            #     predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
        if dist:
            return y_dist, predictions_np
        return predictions_np

    def get_latent(self, batch):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)

        # Run forward pass
        # model = self.node_models_dict[node_type]
        feat_x = self.model.get_latent(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                prediction_horizon=self.ph)
        return feat_x