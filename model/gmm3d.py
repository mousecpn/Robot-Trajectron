import torch
import torch.distributions as td
import numpy as np
from model.model_utils import to_one_hot


class GMM3D(td.Distribution):
    r"""
    Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N,3]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """
    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM3D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:], validate_args=False)
        self.components = log_pis.shape[-1]
        self.dimensions = 3
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = self.reshape_to_components(mus)         # [..., N, 2]
        self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 2]
        self.sigmas = torch.exp(self.log_sigmas)                       # [..., N, 2]
        self.corrs = self.reshape_to_components(corrs)  # [..., N, 3]
        self.one_minus_rho2 = 1 - self.corrs[...,0]**2                        # [..., N]
        self.one_minus_rho2 = torch.clamp(self.one_minus_rho2, min=1e-4, max=1)  # otherwise log can be nan
        self.one_minus_rhos = 1-torch.sum(self.corrs**2,dim=-1)+2*self.corrs[...,0]*self.corrs[...,1]*self.corrs[...,2]
        self.one_minus_rhos = torch.clamp(self.one_minus_rhos, min=1e-4, max=1)


        # self.cov = torch.stack([
        #     torch.stack([self.sigmas[...,0]**2, self.corrs[...,0]*self.sigmas[...,0]*self.sigmas[...,1], self.corrs[...,1]*self.sigmas[...,0]*self.sigmas[...,2]],dim=-1),
        #     torch.stack([self.corrs[...,0]*self.sigmas[...,0]*self.sigmas[...,1], self.sigmas[...,1]**2,  self.corrs[...,2]*self.sigmas[...,1]*self.sigmas[...,2]],dim=-1),
        #     torch.stack([self.corrs[...,1]*self.sigmas[...,0]*self.sigmas[...,2], self.corrs[...,2]*self.sigmas[...,1]*self.sigmas[...,2], self.sigmas[...,2]**2],dim=-1),
        # ], dim=-2)

        # self.cov = self.cov + torch.eye(3).expand(self.cov.shape[0:-2]+(3,3)).to(self.cov.device)*1e-4

        # self.L = torch.cholesky(self.cov, upper=False)

        self.L = torch.stack([torch.stack([self.sigmas[..., 0], torch.zeros_like(self.log_pis), torch.zeros_like(self.log_pis)], dim=-1),
                              torch.stack([self.sigmas[..., 1] * self.corrs[..., 0],
                                           self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2),
                                           torch.zeros_like(self.log_pis)],
                                          dim=-1),
                                torch.stack([self.sigmas[..., 2] * self.corrs[..., 1],
                                    -self.sigmas[..., 2] * (self.corrs[..., 0]*self.corrs[..., 1]-self.corrs[..., 2]) / torch.sqrt(self.one_minus_rho2),
                                    self.sigmas[..., 2] * torch.sqrt(self.one_minus_rhos) / torch.sqrt(self.one_minus_rho2)
                                ],dim=-1)
                                          ],
                             dim=-2)
        

        # self.L = torch.stack([torch.stack([self.sigmas[..., 0], torch.zeros_like(self.log_pis), torch.zeros_like(self.log_pis)], dim=-1),
        #                       torch.stack([torch.zeros_like(self.log_pis),
        #                                    self.sigmas[..., 1],
        #                                    torch.zeros_like(self.log_pis)],
        #                                   dim=-1),
        #                         torch.stack([torch.zeros_like(self.log_pis),
        #                             torch.zeros_like(self.log_pis),
        #                             self.sigmas[..., 2]
        #                         ],dim=-1)
        #                                   ],
        #                      dim=-2)


        size = self.L.shape
        self.cov = torch.bmm(self.L.reshape(-1,self.dimensions,self.dimensions), self.L.reshape(-1,self.dimensions,self.dimensions).permute(0,2,1))
        self.cov = self.cov.reshape(size)

        self.pis_cat_dist = td.Categorical(logits=log_pis)
    

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigma_3 = torch.clamp(cov_mats[..., 2, 2], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2), torch.sqrt(sigma_3)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs1 = cov_mats[..., 0, 1] / sigma_1*sigma_2
        corrs2 = cov_mats[..., 0, 2] / sigma_1*sigma_3
        corrs3 = cov_mats[..., 1, 2] / sigma_2*sigma_3
        corrs = torch.stack((corrs1,corrs2,corrs3),dim=-1)
        return cls(log_pis, mus, log_sigmas, corrs)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = (self.mus +
                       torch.squeeze(
                           torch.matmul(self.L,
                                        torch.unsqueeze(
                                            torch.randn(size=sample_shape + self.mus.shape, device=self.device),
                                            dim=-1)
                                        ),
                           dim=-1))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        if torch.sum(torch.isnan(torch.sum(mvn_samples*selector, dim=-2)))>0:
            print('error')

        return torch.sum(mvn_samples*selector, dim=-2)

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 3]
        dx = value - self.mus                       # [..., N, 3]

        # exp_nominator = ((torch.sum((dx/self.sigmas)**2, dim=-1)  # first and second term of exp nominator
        #                   - 2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)))    # [..., N]

        # component_log_p = -(2*np.log(2*np.pi)
        #                     + torch.log(self.one_minus_rho2)
        #                     + 2*torch.sum(self.log_sigmas, dim=-1)
        #                     + exp_nominator/self.one_minus_rho2) / 2
        
        size = dx.shape
        cov_inv = torch.inverse(self.cov.reshape(-1,self.dimensions, self.dimensions))
        # bs = size[-3]
        # cov_inv = cov_inv.repeat(bs,1,1)
        try:
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dimensions), cov_inv)
        except:
            bs = size[-3]
            cov_inv = cov_inv.repeat(bs,1,1)
            exp_nominator = torch.bmm(dx.reshape(-1,1,self.dimensions), cov_inv)
        exp_nominator = torch.bmm(exp_nominator, dx.reshape(-1,1,self.dimensions).permute(0,2,1))
        exp_nominator = -exp_nominator.reshape(size[0:-1])/2


        component_log_p = - (self.dimensions/2)*np.log(2*np.pi) - (1/2)*torch.det(self.cov) + exp_nominator
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)

    def get_for_node_at_time(self, n, t):
        return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                              self.log_sigmas[:, n:n+1, t:t+1], self.corrs[:, n:n+1, t:t+1])

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min()
                    x_max = self.mus[:, n, t, :, 0].max()
                    y_min = self.mus[:, n, t, :, 1].min()
                    y_max = self.mus[:, n, t, :, 1].max()
                    z_min = self.mus[:, n, t, :, 2].min()
                    z_max = self.mus[:, n, t, :, 2].max()
                    search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max, 0.01),
                                                              torch.arange(y_min, y_max, 0.01),
                                                              torch.arange(z_min, z_max, 0.01)
                                                              ]), dim=2
                                              ).view(-1, 3).float().to(self.device)

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        # cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        # E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov], dim=-1),
        #                  torch.stack([cov, self.sigmas[..., 1]**2], dim=-1)],
        #                 dim=-2)
        return self.cov
