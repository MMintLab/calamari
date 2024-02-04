import torch
class MppiConfig:
    def __init__(self):
        self.dtype = torch.double
        self.device = 'cpu'
        self.action_low = [-0.1, -0.1, -0.001, -0.,-0., -0.4]
        self.action_high = [0.1, 0.1, 0.001, 0., 0., 0.4]
        self.num_samples =  1000

        self.nx = 6
        self.noise_sigma = 0.01 * torch.eye(self.nx, device=self.device, dtype=self.dtype)
        self.noise_sigma[2,2] = 0.001
        self.noise_sigma[5,5] =  0.03
        self.noise_sigma[4,4] = 0.001
        self.noise_sigma[3,3] = 0.001

        self.noise_mu = None

        self.lambda_ = 0.000001 #.001
        self.horizon = 1
        self.eps= 0.05

        self.table_height = 0.750
        self.cnt_eps = 0.01 # 1mm
        self.u_per_command = 5