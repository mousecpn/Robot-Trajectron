import torch
from model.utils import block_diag
from model.gmm3d import GMM3D

class Dynamic(object):
    def __init__(self, dt, dyn_limits, device, xz_size):
        self.dt = dt
        self.device = device
        self.dyn_limits = dyn_limits
        self.initial_conditions = None
        # self.model_registrar = model_registrar
        # self.node_type = node_type
        self.init_constants()
        self.create_graph(xz_size)

    def set_initial_condition(self, init_con):
        self.initial_conditions = init_con

    def init_constants(self):
        pass

    def create_graph(self, xz_size):
        pass

    def integrate_samples(self, s, x):
        raise NotImplementedError

    def integrate_distribution(self, dist, x):
        raise NotImplementedError

    def create_graph(self, xz_size):
        pass

class SingleIntegrator(Dynamic):
    def __init__(self, dt, dyn_limits, device, xz_size):
        super(SingleIntegrator, self).__init__(dt, dyn_limits, device, xz_size)
        return
    
    def init_constants(self):
        self.F = torch.eye(6, device=self.device, dtype=torch.float32)
        self.F[0:3, 3:] = torch.eye(3, device=self.device, dtype=torch.float32) * self.dt
        self.F_t = self.F.transpose(-2, -1)

    def integrate_samples(self, v, x=None):
        """
        Integrates deterministic samples of velocity.

        :param v: Velocity samples
        :param x: Not used for SI.
        :return: Position samples
        """
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        return torch.cumsum(v, dim=2) * self.dt + p_0

    def integrate_distribution(self, v_dist, x=None):
        r"""
        Integrates the GMM velocity distribution to a distribution over position.
        The Kalman Equations are used.

        .. math:: \mu_{t+1} =\textbf{F} \mu_{t}

        .. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

        .. math::
            \textbf{F} = \left[
                            \begin{array}{cccc}
                                \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                                \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                                0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                                0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                            \end{array}
                        \right]_{t}

        :param v_dist: Joint GMM Distribution over velocity in x and y direction.
        :param x: Not used for SI.
        :return: Joint GMM Distribution over position in x and y direction.
        """
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        pos_dist_sigma_matrix_list = []

        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * self.dt

        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(sample_batch_dim + [v_dist.components, 3, 3], device=self.device)

        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = block_diag([pos_dist_sigma_matrix_t, vel_sigma_matrix_t])
            pos_dist_sigma_matrix_t = self.F[..., :3, :].matmul(full_sigma_matrix_t.matmul(self.F_t)[..., :3])
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        return GMM3D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_dist_sigma_matrix)


    def integrate_distribution2zT(self, v_dist, z_T=None):
        r"""
        Integrates the GMM velocity distribution to a distribution over position.
        The Kalman Equations are used.

        .. math:: \mu_{t+1} =\textbf{F} \mu_{t}

        .. math:: \mathbf{\Sigma}_{t+1}={\textbf {F}} \mathbf{\Sigma}_{t} {\textbf {F}}^{T}

        .. math::
            \textbf{F} = \left[
                            \begin{array}{cccc}
                                \sigma_x^2 & \rho_p \sigma_x \sigma_y & 0 & 0 \\
                                \rho_p \sigma_x \sigma_y & \sigma_y^2 & 0 & 0 \\
                                0 & 0 & \sigma_{v_x}^2 & \rho_v \sigma_{v_x} \sigma_{v_y} \\
                                0 & 0 & \rho_v \sigma_{v_x} \sigma_{v_y} & \sigma_{v_y}^2 \\
                            \end{array}
                        \right]_{t}

        :param v_dist: Joint GMM Distribution over velocity in x and y direction.
        :param x: Not used for SI.
        :return: Joint GMM Distribution over position in x and y direction.
        """
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        ph = v_dist.mus.shape[-3]
        sample_batch_dim = list(v_dist.mus.shape[0:2])
        pos_dist_sigma_matrix_list = []
        T = 1

        pos_mus = p_0[:, None] + torch.cumsum(v_dist.mus, dim=2) * self.dt

        vel_dist_sigma_matrix = v_dist.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(sample_batch_dim + [v_dist.components, 3, 3], device=self.device)

        for t in range(ph):
            vel_sigma_matrix_t = vel_dist_sigma_matrix[:, :, t]
            full_sigma_matrix_t = block_diag([pos_dist_sigma_matrix_t, vel_sigma_matrix_t])
            pos_dist_sigma_matrix_t = self.F[..., :3, :].matmul(full_sigma_matrix_t.matmul(self.F_t)[..., :3])
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t)
        
        
        mu_t_minus_1 = pos_mus[:,:,-2:-1]
        mu_t = pos_mus[:,:,-1:]
        dist1 = torch.square(mu_t_minus_1[...,-1].mean() - z_T)
        dist2 = torch.square(mu_t[...,-1].mean() - z_T)
        weight1 = torch.exp(-dist1/T)/(torch.exp(-dist1/T)+torch.exp(-dist2/T))
        weight2 = torch.exp(-dist2/T)/(torch.exp(-dist1/T)+torch.exp(-dist2/T))
        mu_z_T = weight1*mu_t_minus_1 + weight2*mu_t
        pos_dist_sigma_matrix_z_T = weight1*pos_dist_sigma_matrix_list[-2] + weight2*pos_dist_sigma_matrix_list[-1]
        pos_mus[:,:,-1:] = mu_z_T
        pos_dist_sigma_matrix_list[-1] = pos_dist_sigma_matrix_z_T

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        return GMM3D.from_log_pis_mus_cov_mats(v_dist.log_pis, pos_mus, pos_dist_sigma_matrix)



