import os

# import torchcde
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn

# from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as dto

# from TorchDiffEqPack.odesolver import ode_solver
import flow as flows
import sim_config
from global_config import DTYPE, get_device


class GaussianReparam:
    """
    Independent Gaussian posterior with re-parameterization trick
    """

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def log_density(mu, log_var, z):
        n = dist.normal.Normal(mu, torch.exp(0.5 * log_var))
        log_p = torch.sum(n.log_prob(z), dim=-1)
        return log_p


class StandardNormalPrior:
    @staticmethod
    def log_density(z):
        n = dist.normal.Normal(torch.tensor([0.0]).to(z), torch.tensor([1.0]).to(z))
        return torch.sum(n.log_prob(z), dim=-1)


class ExponentialPrior:
    @staticmethod
    def log_density(z):
        n = dist.exponential.Exponential(rate=torch.tensor([100.0]).to(z))
        return torch.sum(n.log_prob(z), dim=-1)


class EncoderPlanarLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_flows, normalize=True, device=None):
        super().__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.model_name = "PlanarLSTMEncoder"

        self.lstm = nn.LSTM(input_dim, hidden_dim).to(self.device)

        self.lin = nn.Linear(hidden_dim, output_dim).to(self.device)

        self.log_var = nn.Linear(hidden_dim, output_dim).to(self.device)

        self.q_z_nn_output_dim = hidden_dim
        self.num_flows = num_flows
        self.z_size = output_dim

        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.0

        # Flow parameters
        flow = flows.Planar
        self.num_flows = num_flows

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size).to(self.device)
        self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size).to(self.device)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows).to(self.device)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow().to(self.device)
            self.add_module("flow_" + str(k), flow_k)

    def forward(self, x, a, mask):
        # y and t are the first k observations
        batch_size = x.shape[1]
        t_max = x.shape[0]
        x = x.squeeze()

        y_in = torch.cat([x, a], dim=-1)
        mask_in = torch.cat([mask, torch.ones_like(a)], dim=-1)

        hidden = None

        for t in reversed(range(t_max)):
            obs = y_in[t : t + 1, ...] * mask_in[t : t + 1, ...]
            out, hidden = self.lstm(obs, hidden)

        out_linear = self.lin(out)
        log_var = self.log_var(out)

        # B, n_flow, D, 1
        u = self.amor_u(out)[0, ...].view(batch_size, self.num_flows, self.z_size, 1)
        # B, n_flow, 1, D
        w = self.amor_w(out)[0, ...].view(batch_size, self.num_flows, 1, self.z_size)
        # B, n_flow, 1, 1
        b = self.amor_b(out)[0, ...].view(batch_size, self.num_flows, 1, 1)

        # B, D
        mu = out_linear[0, ...]
        log_var = log_var[0, ...]

        if self.normalize:
            # scale mu
            mu = torch.exp(mu) / 10
            # mask = torch.zeros_like(mu)
            # mask[:, 0] = 1
            # mu = mu * mask

            # scale var
            log_var = log_var - 5.0

        return mu, log_var, u, w, b

    def reparameterize(self, mu, log_var, u, w, b):

        log_det_j = 0.0

        # Sample z_0
        z = [GaussianReparam.reparameterize(mu, log_var)]

        # Normalizing flows
        for k in range(self.num_flows):
            flow_k = getattr(self, "flow_" + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            log_det_j += log_det_jacobian

        # todo: add one normalization layer
        z_exp = torch.exp(z_k - 5.0)
        z.append(z_exp)
        log_det_j += torch.sum(z_k - 5.0, dim=-1)

        return mu, log_var, z[-1], log_det_j, z[0]

    def log_density(self, mu, log_var, z_1, log_det_j, z0):

        q_z0 = GaussianReparam.log_density(mu, log_var, z0)

        return q_z0 - log_det_j


class F(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(F, self).__init__()
        input_channels = input_dim
        hidden_channels = hidden_dim

        self.input_channels = input_dim
        self.hidden_channels = hidden_dim

        # For illustrative purposes only. You should usually use an MLP or something. A single linear layer won't be
        # that great.
        #                 self.linear = torch.nn.Linear(hidden_channels,
        #                                       hidden_channels * input_channels)
        self.linear = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * input_channels + 1),
            nn.Tanh(),
            nn.Linear(hidden_channels * input_channels + 1, hidden_channels * input_channels),
        )

    def forward(self, t, z):
        batch_dims = z.shape[:-1]
        return self.linear(z).tanh().view(*batch_dims, self.hidden_channels, self.input_channels)


class EncoderLSTMReal(nn.Module, GaussianReparam):
    def __init__(self, input_dim, hidden_dim, output_dim, output_all=False, reverse=True, normalize=True, device=None):
        super(EncoderLSTMReal, self).__init__()

        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.output_dim = output_dim
        self.model_name = "LSTMReal"

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim).to(self.device)

        # The linear layer that maps from hidden state space to output space: predict mean
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim + 1), nn.Tanh(), nn.Linear(hidden_dim + 1, output_dim), nn.Tanh(),
        ).to(self.device)

        self.log_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim + 1), nn.Tanh(), nn.Linear(hidden_dim + 1, output_dim), nn.Tanh(),
        ).to(self.device)

        self.reverse = reverse
        self.output_all = output_all

    def forward(self, x, a, m):
        if self.reverse:
            x = torch.flip(x, [0])
            a = torch.flip(a, [0])
            m = torch.flip(m, [0])

        t = torch.arange(m.shape[0])
        t = torch.stack([t] * m.shape[1], dim=-1)
        t = t[..., None]
        t = t / m.max()
        x_in = torch.cat([x, a, t], dim=-1)

        t_max = x_in.shape[0]

        hidden = None
        out_list = list()
        for t in range(t_max):
            obs = x_in[t : t + 1, ...]
            out, hidden = self.lstm(obs, hidden)
            out_list.append(out)

        out = torch.cat(out_list, dim=0)
        out_linear = self.lin(out)
        log_var = self.log_var(out)

        # B, D
        if self.output_all:
            return out_linear, log_var
        else:

            mu = out_linear[-1, ...]
            log_var = log_var[-1, ...]
            return mu, log_var


# class CDE(nn.Module, GaussianReparam):
#     def __init__(self, input_dim, hidden_dim, output_dim, method, ode_step_size, normalize=True, device=None):
#         super(CDE, self).__init__()
#
#         if device is None:
#             self.device = get_device()
#         else:
#             self.device = device
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.normalize = normalize
#         self.output_dim = output_dim
#         self.model_name = 'CDE'
#
#         input_channels = input_dim
#         hidden_channels = hidden_dim
#         output_channels = output_dim
#
#         self.initial = nn.Linear(input_channels, hidden_channels)
#         self.func = F(input_dim, hidden_dim)
#
#         self.readout = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels + 1),
#             nn.Tanh(),
#             nn.Linear(hidden_channels + 1, output_channels),
#             nn.Tanh(),
#         )
#
#         self.readout2 = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels + 1),
#             nn.Tanh(),
#             nn.Linear(hidden_channels + 1, output_channels),
#             nn.Tanh(),
#         )
#
#         # self.readout = nn.Linear(hidden_channels, output_channels)
#         # self.readout2 = nn.Linear(hidden_channels, output_channels)
#
#         options = {}
#         # options.update({'step_t': self.t})
#         # options.update({'jump_t': self.t})
#         options.update({'step_size': ode_step_size})
#         options.update({'perturb': True})
#         self.rtol = 1e-7
#         self.atol = 1e-8
#         self.options = options
#         self.method = method
#         self.step_size = ode_step_size
#
#     def forward(self, x, a, m):
#         x = torch.flip(x, [0])
#         a = torch.flip(a, [0])
#         m = torch.flip(m, [0])
#         m = m == 1
#         x[~m] = torch.tensor(np.nan)
#         m = torch.cumsum(m, dim=0)
#         t = torch.arange(m.shape[0])
#         t = torch.stack([t] * m.shape[1], dim=-1)
#         t = t[..., None]
#         x_in = torch.cat([x, a, m, t], dim=-1)
#         x = x_in.permute((1, 0, 2))
#
#         coeffs = torchcde.natural_cubic_coeffs(x)
#         X = torchcde.NaturalCubicSpline(coeffs)
#         X0 = X.evaluate(X.interval[0])
#         assert torch.sum(torch.isnan(X0)) == 0
#         z0 = self.initial(X0)
#         zt = torchcde.cdeint(X=X, func=self.func, z0=z0, t=X.interval, adjoint=False, method=self.method, options=self.options)
#         zT = zt[..., -1, :]  # get the terminal value of the CDE
#         assert torch.sum(torch.isnan(zT)) == 0
#
#         mu = self.readout(zT)
#         log_sig = self.readout2(zT)
#
#         return mu, log_sig


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True, device=None):
        super().__init__()

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.output_dim = output_dim
        self.model_name = "LSTMBaseline"

        self.lstm = nn.LSTM(input_dim, hidden_dim).to(self.device)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim + 1, bias=True),
            nn.ELU(),
            nn.Linear(self.hidden_dim + 1, self.output_dim, bias=True),
        ).to(self.device)

    def forward(self, x, a, mask):
        # y and t are the first k observations

        t_max = x.shape[0]

        x = x.squeeze()

        y_in = torch.cat([x, a], dim=-1)

        hidden = None
        out, hidden = self.lstm(y_in, hidden)

        pred = self.out(out)
        return pred

    def loss(self, data):
        x = data["measurements"]
        a = data["actions"]
        mask = data["masks"]
        s = data["statics"]

        # q
        a_in = torch.cat([a, s], dim=-1)
        x_hat = self.forward(x, a_in, mask)[:-1]

        # average over B (samples in mini batch)
        # todo off set
        lik = torch.sum((x[1:] - x_hat) ** 2 * mask[1:]) / x.shape[1]

        return lik

    def save(self, path, itr, best_loss):

        path = path + self.model_name
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({"itr": itr, "state_dict": self.state_dict(), "best_loss": best_loss,}, path)


class EncoderLSTM(nn.Module, GaussianReparam):
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True, device=None):
        # output dim is the dim of initial condition
        # input dim is observation and action

        super(EncoderLSTM, self).__init__()

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.model_name = "LSTMEncoder"

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim, hidden_dim).to(self.device)

        # The linear layer that maps from hidden state space to output space: predict mean
        self.lin = nn.Linear(hidden_dim, output_dim).to(self.device)

        self.log_var = nn.Linear(hidden_dim, output_dim).to(self.device)

    def forward(self, x, a, mask):
        # y and t are the first k observations

        t_max = x.shape[0]

        x = x.squeeze()

        y_in = torch.cat([x, a], dim=-1)
        mask_in = torch.cat([mask, torch.ones_like(a)], dim=-1)

        hidden = None

        for t in reversed(range(t_max)):
            obs = y_in[t : t + 1, ...] * mask_in[t : t + 1, ...]
            out, hidden = self.lstm(obs, hidden)
        out_linear = self.lin(out)
        log_var = self.log_var(out)

        # B, D
        mu = out_linear[0, ...]
        log_var = log_var[0, ...]

        if self.normalize:
            # scale mu
            mu = torch.exp(mu) / 10
            # mask = torch.zeros_like(mu)
            # mask[:, 0] = 1
            # mu = mu * mask

            # scale var
            log_var = log_var - 5.0

        return mu, log_var

    # todo: use a distribution with non-negative support
    # def reparameterize(self, mu, log_var):


class RocheODE(nn.Module):
    def __init__(self, latent_dim, action_dim, t_max, step_size, ablate=False, device=None, dtype=DTYPE):
        super().__init__()

        assert action_dim == 1

        self.action_dim = action_dim
        self.latent_dim = int(latent_dim)
        self.expert_dim = int(4)
        self.ml_dim = self.latent_dim - self.expert_dim
        self.expanded = True if self.ml_dim > 0 else False
        self.ablate = ablate

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t_max = t_max
        self.step_size = step_size

        dc = sim_config.RochConfig()
        self.HillCure = nn.Parameter(torch.tensor(dc.HillCure, device=self.device, dtype=dtype))
        self.HillPatho = nn.Parameter(torch.tensor(dc.HillPatho, device=self.device, dtype=dtype))
        self.ec50_patho = nn.Parameter(torch.tensor(dc.ec50_patho, device=self.device, dtype=dtype))
        self.emax_patho = nn.Parameter(torch.tensor(dc.emax_patho, device=self.device, dtype=dtype))
        self.k_dexa = nn.Parameter(torch.tensor(dc.k_dexa, device=self.device, dtype=dtype))
        self.k_discure_immunereact = nn.Parameter(
            torch.tensor(dc.k_discure_immunereact, device=self.device, dtype=dtype)
        )
        self.k_discure_immunity = nn.Parameter(torch.tensor(dc.k_discure_immunity, device=self.device, dtype=dtype))
        self.k_disprog = nn.Parameter(torch.tensor(dc.k_disprog, device=self.device, dtype=dtype))
        self.k_immune_disease = nn.Parameter(torch.tensor(dc.k_immune_disease, device=self.device, dtype=dtype))
        self.k_immune_feedback = nn.Parameter(torch.tensor(dc.k_immune_feedback, device=self.device, dtype=dtype))
        self.k_immune_off = nn.Parameter(torch.tensor(dc.k_immune_off, device=self.device, dtype=dtype))
        self.k_immunity = nn.Parameter(torch.tensor(dc.k_immunity, device=self.device, dtype=dtype))
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=dtype))
        if self.ablate:
            self.theta_1 = nn.Parameter(torch.tensor(1, device=self.device, dtype=dtype))
            self.theta_2 = nn.Parameter(torch.tensor(2, device=self.device, dtype=dtype))

        if self.expanded:
            self.ml_net = nn.Sequential(nn.Linear(self.latent_dim, self.ml_dim), nn.Tanh()).to(self.device)
        else:
            self.ml_net = nn.Identity().to(self.device)

        self.times = None
        self.dosage = None

    def set_action(self, action):
        # T, B, D
        # B
        self.dosage = torch.max(action[..., 0], dim=0)[0]

        time_list = []
        for i in range(action.shape[1]):
            time = torch.where(action[..., 0][:, i] != 0)[0]
            time = time * self.step_size
            time_list.append(time)

        # B, N_DOSE
        self.times = torch.stack(time_list, dim=0)

    def dose_at_time(self, t):
        # self.t = t
        return self.dosage * torch.sum(
            torch.exp(self.kel * (self.times - t) * (t >= self.times)) * (t >= self.times), dim=-1
        )

    def forward(self, t, y):
        # y: B, D

        # B
        Disease = y[:, 0]
        ImmuneReact = y[:, 1]
        Immunity = y[:, 2]
        Dose2 = y[:, 3]

        if not self.ablate:
            Dose = self.dose_at_time(t)

            dxdt1 = (
                Disease * self.k_disprog
                - Disease * Immunity ** self.HillCure * self.k_discure_immunity
                - Disease * ImmuneReact * self.k_discure_immunereact
            )

            dxdt2 = (
                Disease * self.k_immune_disease
                - ImmuneReact * self.k_immune_off
                + Disease * ImmuneReact * self.k_immune_feedback
                + (ImmuneReact ** self.HillPatho * self.emax_patho)
                / (self.ec50_patho ** self.HillPatho + ImmuneReact ** self.HillPatho)
                - Dose2 * ImmuneReact * self.k_dexa
            )

            dxdt3 = ImmuneReact * self.k_immunity

            dxdt4 = self.kel * Dose - self.kel * Dose2
        else:
            dxdt1 = ImmuneReact
            dxdt2 = -1.0 * Disease * self.theta_1
            dxdt3 = Dose2
            dxdt4 = -1.0 * Immunity * self.theta_2

        if self.expanded:
            dmldt = self.ml_net(y)
            return torch.cat([dxdt1[..., None], dxdt2[..., None], dxdt3[..., None], dxdt4[..., None], dmldt], dim=-1)
        else:
            return torch.stack([dxdt1, dxdt2, dxdt3, dxdt4], dim=-1)


class MaskedLinear(nn.Module):
    def __init__(self, input_dim, mask):
        super().__init__()
        self.input_dim = input_dim
        self.mask = mask

        self.net = nn.Parameter(torch.randn(input_dim, input_dim) / np.sqrt(input_dim))

    def forward(self, mat):
        return torch.matmul(mat, self.mask * self.net)


class RocheODEReal(nn.Module):
    def __init__(self, latent_dim, action_dim, static_dim, hidden_dim, t_max, step_size, device=None, dtype=DTYPE):
        super().__init__()
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)
        self.dosage = None
        self.times = None

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t_max = t_max
        self.step_size = step_size

        self.dx1_net = nn.Sequential(nn.Linear(3, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, 1), nn.Tanh())

        self.dx2_net = nn.Sequential(nn.Linear(2, self.hidden_dim), nn.Tanh(), nn.Linear(self.hidden_dim, 1), nn.Tanh())

        self.expert_dim = 4

        if self.latent_dim == self.expert_dim:
            self.expert_only = True
        else:
            self.expert_only = False
        if not self.expert_only:
            self.lin_hh = torch.nn.Linear(
                self.latent_dim - self.expert_dim, self.latent_dim - self.expert_dim, bias=False
            )
            self.lin_hz = torch.nn.Linear(
                self.latent_dim - self.expert_dim, self.latent_dim - self.expert_dim, bias=False
            )
            self.lin_hr = torch.nn.Linear(
                self.latent_dim - self.expert_dim, self.latent_dim - self.expert_dim, bias=False
            )

        self.k_immunity = nn.Parameter(torch.tensor(1, device=self.device, dtype=dtype))
        self.kel = nn.Parameter(torch.tensor(0.2, device=self.device, dtype=dtype))
        self.kel2 = nn.Parameter(torch.tensor(0.2, device=self.device, dtype=dtype))

    def forward(self, t, y):
        # y: B, D
        Disease = y[:, 0]
        ImmuneReact = y[:, 1]
        # Immunity = y[:, 2]
        Dose2 = y[:, 3]
        Dose = self.dose_at_time(t)

        # Disease
        dx1_in = y[:, :3]
        dxdt1 = self.dx1_net(dx1_in)

        dx2_in = y[:, :2]
        dxdt2 = self.dx2_net(dx2_in)

        dxdt3 = (ImmuneReact * self.k_immunity)[..., None]

        dxdt4 = (self.kel * Dose - self.kel2 * Dose2)[..., None]

        if self.expert_only:
            ret = torch.cat([dxdt1, dxdt2, dxdt3, dxdt4], dim=-1)
            return ret
        else:
            x = 0
            h = y[..., self.expert_dim :]
            r = torch.sigmoid(x + self.lin_hr(h))
            z = torch.sigmoid(x + self.lin_hz(h))
            u = torch.tanh(x + self.lin_hh(r * h))

            dmldt = (1 - z) * (u - h)

            ret = torch.cat([dxdt1, dxdt2, dxdt3, dxdt4, dmldt], dim=-1)
            return ret

    def set_action_static(self, action, static):
        # T B 1
        self.dosage = action
        # T B 1
        self.times = torch.cumsum(torch.ones_like(action), dim=0)

    def dose_at_time(self, t):
        # assert self.dosage.shape[-1] == 1
        inside_exp = self.kel * (self.times - t) * (t >= self.times)
        d = torch.sum(self.dosage * torch.exp(inside_exp) * (t >= self.times), dim=(0, 2))
        return d


class NeuralODEReal2nd(nn.Module):
    def __init__(self, latent_dim, action_dim, static_dim, hidden_dim, t_max, step_size, device=None, dtype=DTYPE):
        super().__init__()
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t_max = t_max
        self.step_size = step_size

        self.ml_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.latent_dim // 2),
            nn.Tanh(),
        ).to(self.device)

        self.action = None
        self.static = None

    def set_action_static(self, action, static):
        # T B D
        self.action = action
        # B D
        self.static = static[0, :, :]

    def dose_at_time(self, t):
        # todo: remove nan
        # if torch.sum(torch.isnan(t)) > 0:
        #     return torch.zeros_like(self.action[0, :, :])
        t_int = int(t)

        t_max = self.action.shape[0]

        if t_int >= t_max:
            return torch.zeros_like(self.action[0, :, :])
        else:
            return torch.cumsum(self.action, dim=0)[t_int, :, :]

    def forward(self, t, y):
        # y: B, D

        dose = self.dose_at_time(t)
        y_full = torch.cat([y, dose], dim=-1)

        dml1dt = self.ml_net(y_full)
        dml2dt = y[..., : (self.latent_dim // 2)]
        dmldt = torch.cat([dml1dt, dml2dt], dim=-1)
        return dmldt


class NeuralODEReal(nn.Module):
    def __init__(self, latent_dim, action_dim, static_dim, hidden_dim, t_max, step_size, device=None, dtype=DTYPE):
        super().__init__()
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t_max = t_max
        self.step_size = step_size

        self.ml_net = nn.Sequential(
            nn.Linear(self.latent_dim + self.action_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.Tanh(),
        ).to(self.device)

        self.action = None
        self.static = None

    def set_action_static(self, action, static):
        # T B D
        self.action = action
        # B D
        self.static = static[0, :, :]

    def dose_at_time(self, t):
        # todo: remove nan
        # if torch.sum(torch.isnan(t)) > 0:
        #     return torch.zeros_like(self.action[0, :, :])
        t_int = int(t)

        t_max = self.action.shape[0]

        if t_int >= t_max:
            return torch.zeros_like(self.action[0, :, :])
        else:
            return torch.cumsum(self.action, dim=0)[t_int, :, :]

    def forward(self, t, y):
        # y: B, D

        dose = self.dose_at_time(t)
        y_full = torch.cat([y, dose], dim=-1)

        dmldt = self.ml_net(y_full)
        return dmldt


class DecoderReal(nn.Module):
    def __init__(
        self,
        obs_dim,
        latent_dim,
        action_dim,
        static_dim,
        hidden_dim,
        t_max,
        step_size,
        t0=0,
        method="dopri5",
        ode_step_size=None,
        ode_type="neural",
        device=None,
        dtype=DTYPE,
    ):
        super().__init__()

        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.t0 = t0
        self.step_size = step_size
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        # todo: switch between decoders
        self.model_name = "DecoderReal_" + ode_type

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.output_function = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim + 1, bias=True),
            nn.ELU(),
            nn.Linear(self.latent_dim + 1, self.obs_dim, bias=True),
        ).to(self.device)

        if ode_type == "neural":
            self.ode = NeuralODEReal(latent_dim, action_dim, static_dim, hidden_dim, t_max, step_size, device)
        elif ode_type == "2nd":
            self.ode = NeuralODEReal2nd(latent_dim, action_dim, static_dim, hidden_dim, t_max, step_size, device)
        else:
            self.ode = RocheODEReal(latent_dim, action_dim, static_dim, hidden_dim, t_max, step_size, device)
        self.t = torch.arange(t0 - 1, t_max, step_size, device=self.device, dtype=dtype)
        options = {}
        options.update({"step_t": self.t})
        # options.update({'jump_t': self.t})
        options.update({"step_size": ode_step_size})
        options.update({"perturb": True})
        self.rtol = 1e-7
        self.atol = 1e-8
        self.options = options
        self.method = method
        self.step_size = ode_step_size

    def forward(self, init, a, s):
        self.ode.set_action_static(a, s)
        if len(init.shape) == 2:
            # solve ode
            h = dto(self.ode, init, self.t, method=self.method, options=self.options, rtol=self.rtol, atol=self.atol)
        else:
            h_list = []
            for i in range(self.t_max - 1):
                ht0 = init[i]
                ht = dto(
                    self.ode,
                    ht0,
                    self.t[i : (i + 2)],
                    method=self.method,
                    options=self.options,
                    rtol=self.rtol,
                    atol=self.atol,
                )
                h_list.append(ht[-1, ...])

            padding = torch.zeros_like(h_list[0])
            # print('padding', padding.shape)
            h_list = [padding] + h_list
            h = torch.stack(h_list, dim=0)
            # print('h', h.shape)
        # generate output
        x_hat = self.output_function(h)[1:]
        if len(init.shape) != 2:
            x_hat[0] = 0.0
        return x_hat, h


class GRUODECell(torch.nn.Module):
    """
    https://github.com/edebrouwer/gru_ode_bayes/blob/master/gru_ode_bayes/models.py
    """

    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.lin_hz = torch.nn.Linear(hidden_size + 2, hidden_size + 2, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size + 2, hidden_size, bias=False)

    def forward(self, a, h_all):

        h = torch.cat([h_all[0], a], dim=-1)

        z = torch.sigmoid(self.lin_hz(h))
        n = torch.tanh(self.lin_hn(z * h))

        # print(z.shape)
        # print(n.shape)
        # print(h_all[0].shape)
        dh = (1 - z[:, :, : self.hidden_size]) * (n - h_all[0])
        return dh, (h_all[0], 0)


class DecoderRealBenchmark(nn.Module):
    def __init__(
        self,
        obs_dim,
        latent_dim,
        action_dim,
        static_dim,
        hidden_dim,
        t_max,
        step_size,
        t0=0,
        method="dopri5",
        ode_step_size=None,
        ode_type="tlstm",
        device=None,
        dtype=DTYPE,
    ):
        super().__init__()

        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.t0 = t0
        self.step_size = step_size
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

        # todo: switch between decoders
        self.model_name = "DecoderReal_" + ode_type

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.output_function = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim + 1, bias=True),
            nn.ELU(),
            nn.Linear(self.latent_dim + 1, self.obs_dim, bias=True),
        ).to(self.device)
        self.ode_type = ode_type

        if ode_type == "tlstm":
            self.rnn = nn.LSTM(action_dim * 2, latent_dim).to(self.device)
        elif ode_type == "gruode":
            self.rnn = GRUODECell(latent_dim)
            # self.rnn = nn.GRUCell(action_dim * 2, latent_dim).to(self.device)
        self.t = torch.arange(t0, t_max, step_size, device=self.device, dtype=dtype)
        self.method = method
        self.step_size = ode_step_size

    def forward(self, init, a, s):
        if self.ode_type in ["tlstm", "gruode"]:
            hidden = init[None, :, :]
        else:
            hidden = init
        c = init[None, :, :]

        out_list = list()
        for tt in self.t:
            t = int(tt.item())
            obs = a[t : t + 1, ...]
            time = torch.ones_like(obs) * t / self.t_max
            obs = torch.cat([obs, time], dim=-1)
            if self.ode_type in ["tlstm", "gruode"]:
                out, (hidden, c) = self.rnn(obs, (hidden, c))
            else:
                hidden = self.rnn(obs[0], hidden)
                out = hidden[None, ...]
            out_list.append(out)

        h = torch.cat(out_list, dim=0)

        # generate output
        x_hat = self.output_function(h)
        return x_hat, h


class NeuralODE(nn.Module):
    def __init__(self, latent_dim, action_dim, t_max, step_size, device=None, dtype=DTYPE):
        super().__init__()

        assert action_dim == 1

        self.action_dim = action_dim
        self.latent_dim = int(latent_dim)
        self.expert_dim = int(4)
        self.ml_dim = self.latent_dim

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t_max = t_max
        self.step_size = step_size

        dc = sim_config.RochConfig()
        self.kel = nn.Parameter(torch.tensor(dc.kel, device=self.device, dtype=dtype))

        self.ml_net = nn.Sequential(
            nn.Linear(self.latent_dim + 1, self.latent_dim * 10),
            nn.Tanh(),
            nn.Linear(self.latent_dim * 10, self.latent_dim),
            nn.Tanh(),
        ).to(self.device)

        self.times = None
        self.dosage = None

    def set_action(self, action):
        # T, B, D
        # B
        self.dosage = torch.max(action[..., 0], dim=0)[0]

        time_list = []
        for i in range(action.shape[1]):
            time = torch.where(action[..., 0][:, i] != 0)[0]
            time = time * self.step_size
            time_list.append(time)

        # B, N_DOSE
        self.times = torch.stack(time_list, dim=0)

    def dose_at_time(self, t):
        # self.t = t
        return self.dosage * torch.sum(self.times == t, dim=-1)

    def forward(self, t, y):
        # y: B, D

        Dose = self.dose_at_time(t)
        y_full = torch.cat([y, Dose[:, None]], dim=-1)

        dmldt = self.ml_net(y_full)
        return dmldt


# assume all outputs are tanh [-1, 1]
class RocheExpertDecoder(nn.Module):
    def __init__(
        self,
        obs_dim,
        latent_dim,
        action_dim,
        t_max,
        step_size,
        roche=True,
        ablate=False,
        method="dopri5",
        ode_step_size=None,
        device=None,
        dtype=DTYPE,
    ):
        super().__init__()

        self.time_dim = int(t_max / step_size)
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.t_max = t_max
        self.step_size = step_size
        self.roche = roche
        self.ablate = ablate
        if roche:
            if latent_dim == 4:
                self.model_name = "ExpertDecoder"
            else:
                self.model_name = "HybridDecoder"
        else:
            self.model_name = "NeuralODEDecoder"

        if self.ablate:
            self.model_name = self.model_name + "Ablate"
            print("Running ablation study")

        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.t = torch.arange(0, t_max + step_size, step_size, device=self.device, dtype=dtype)

        options = {}
        options.update({"method": method})
        options.update({"h": ode_step_size})
        options.update({"t0": 0.0})
        options.update({"t1": t_max + step_size})
        options.update({"rtol": 1e-7})
        options.update({"atol": 1e-8})
        options.update({"print_neval": True})
        options.update({"neval_max": 1000000})
        options.update({"safety": None})
        options.update({"t_eval": self.t})
        options.update({"interpolation_method": "cubic"})
        options.update({"regenerate_graph": False})

        self.options = options

        # self.output_function = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim * 2),
        #     nn.ELU(),
        #     nn.Linear(self.latent_dim * 2, self.obs_dim),
        #     nn.Tanh()
        # ).to(self.device)

        self.output_function = nn.Sequential(
            nn.Linear(self.latent_dim, self.obs_dim, bias=True),
            # nn.Tanh()
        ).to(self.device)

        # note: debug only
        # self.output_function = nn.Sequential(
        #     nn.Identity()
        # nn.Tanh()
        # ).to(self.device)
        if roche:
            self.ode = RocheODE(latent_dim, action_dim, t_max, step_size, ablate=self.ablate, device=device)
        else:
            self.ode = NeuralODE(latent_dim, action_dim, t_max, step_size, device)

    def forward(self, init, a):
        self.ode.set_action(a)
        # solve ode
        # h = ode_solver.odesolve(self.ode, init, self.options)
        h = dto(
            self.ode, init, self.t, rtol=self.options["rtol"], atol=self.options["atol"], method=self.options["method"]
        )
        # generate output
        x_hat = self.output_function(h)
        return x_hat, h


class VariationalInference:
    def __init__(self, encoder, decoder, elbo=True, prior_log_pdf=None, mc_size=100):
        self.encoder = encoder
        self.decoder = decoder
        self.prior_log_pdf = prior_log_pdf
        self.mc_size = mc_size
        self.elbo = elbo
        self.model_name = "VI_{}_{}.pkl".format(encoder.model_name, decoder.model_name)

    def save(self, path, itr, best_loss):

        path = path + self.model_name
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "itr": itr,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "best_loss": best_loss,
            },
            path,
        )

    def loss(self, data):
        x = data["measurements"]
        a = data["actions"]
        mask = data["masks"]

        self.x = x
        self.a = a
        self.mask = mask

        # q
        mu, log_var = self.encoder(x, a, mask)

        self.mu = mu
        self.log_var = log_var

        # B, D
        if self.elbo:
            z = self.encoder.reparameterize(mu, log_var)
        else:
            z = mu
        self.z = z

        x_hat, h_hat = self.decoder(z, a)

        self.x_hat = x_hat
        self.h_hat = h_hat

        # todo: likelihood loss - using MSE for now
        # average over B (samples in mini batch)
        lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]

        if not self.elbo:
            return lik

        # KL loss
        # TODO: assume standard normal prior

        if self.prior_log_pdf is None:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        else:
            kld_loss = torch.mean(self.mc_kl(mu, log_var, self.mc_size), dim=0)

        loss = lik + kld_loss
        return loss

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def mc_kl(self, mu, log_var, sample_size):
        mc_samples = list()

        for i in range(sample_size):
            # sample from q(z)
            z = self.encoder.reparameterize(mu, log_var)
            # todo
            z[z < 0] = 1e-9
            # log p(z)
            log_p = self.prior_log_pdf(z)
            # log q(z)
            log_q = self.encoder.log_density(mu, log_var, z)
            mc_samples.append(log_q - log_p)

        mc_tensor = torch.stack(mc_samples, dim=-1)
        mc_mean = torch.mean(mc_tensor, dim=-1)
        return mc_mean


class VariationalInferenceReal(VariationalInference):
    def __init__(self, encoder, decoder, elbo=True, prior_log_pdf=None, mc_size=100, t0=24, weight=False):
        super().__init__(encoder, decoder, elbo, prior_log_pdf, mc_size)
        self.t0 = t0
        self.weight = weight

    def loss(self, data):
        x = data["measurements"]
        a = data["actions"]
        mask = data["masks"]
        s = data["statics"]
        t0 = self.t0

        # q
        a_in = torch.cat([a, s], dim=-1)
        mu, log_var = self.encoder(x[:t0], a_in[:t0], mask[:t0])

        # B, D
        if self.elbo:
            z = self.encoder.reparameterize(mu, log_var)
        else:
            z = mu

        x_hat, h_hat = self.decoder(z, a, s)

        # average over B (samples in mini batch)
        if self.weight:
            weight = 1 / torch.arange(1, self.decoder.t_max - t0 + 1)[:, None, None]
        else:
            weight = 1.0
        lik = torch.sum((x[t0:] - x_hat) ** 2 * mask[t0:] * weight) / x[t0:].shape[1]

        if not self.elbo:
            return lik

        # KL loss
        if len(log_var.shape) == 2:
            if self.prior_log_pdf is None:
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            else:
                kld_loss = torch.mean(self.mc_kl(mu, log_var, self.mc_size), dim=0)
        else:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))
        loss = lik + kld_loss
        return loss

    # def loss(self, data):
    #     x = data['measurements']
    #     a = data['actions']
    #     mask = data['masks']
    #     s = data['statics']
    #
    #     # q
    #     a_in = torch.cat([a, s], dim=-1)
    #     mu, log_var = self.encoder(x, a_in, mask)
    #
    #     # B, D
    #     if self.elbo:
    #         z = self.encoder.reparameterize(mu, log_var)
    #     else:
    #         z = mu
    #
    #     x_hat, h_hat = self.decoder(z, a, s)
    #
    #     # average over B (samples in mini batch)
    #     lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]
    #
    #     if not self.elbo:
    #         return lik
    #
    #     # KL loss
    #     if len(log_var.shape) == 2:
    #         if self.prior_log_pdf is None:
    #             kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    #         else:
    #             kld_loss = torch.mean(self.mc_kl(mu, log_var, self.mc_size), dim=0)
    #     else:
    #         kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))
    #     loss = lik + kld_loss
    #     return loss


class VariationalInferenceFlow:
    def __init__(self, encoder, decoder, elbo=True, prior_log_pdf=None, mc_size=100):
        self.encoder = encoder
        self.decoder = decoder
        self.prior_log_pdf = prior_log_pdf
        self.mc_size = mc_size
        self.elbo = elbo
        self.model_name = "VI_FLOW_{}_{}.pkl".format(encoder.model_name, decoder.model_name)

    def save(self, path, itr, best_loss):

        path = path + self.model_name
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "itr": itr,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
                "best_loss": best_loss,
            },
            path,
        )

    def loss(self, data):
        x = data["measurements"]
        a = data["actions"]
        mask = data["masks"]

        self.x = x
        self.a = a
        self.mask = mask

        # q
        encoder_out = self.encoder(x, a, mask)

        # B, D
        mu, log_var, z, log_det_j, z0 = self.encoder.reparameterize(*encoder_out)

        self.z = z

        x_hat, h_hat = self.decoder(z, a)

        self.x_hat = x_hat
        self.h_hat = h_hat

        # average over B (samples in mini batch)
        lik = torch.sum((x - x_hat) ** 2 * mask) / x.shape[1]

        # KL loss

        if self.mc_size == 1:
            log_p = self.prior_log_pdf(z)
            log_q = self.encoder.log_density(mu, log_var, z, log_det_j, z0)
            kld_loss = torch.mean(log_p - log_q, dim=0)
        else:
            kld_loss = torch.mean(self.mc_kl(encoder_out, self.mc_size), dim=0)

        loss = lik + kld_loss
        if self.elbo:
            return loss
        else:
            return lik

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def mc_kl(self, encoder_out, sample_size):
        mc_samples = list()

        for i in range(sample_size):
            # sample from q(z)
            mu, log_var, z, log_det_j, z0 = self.encoder.reparameterize(*encoder_out)
            # log p(z)
            log_p = self.prior_log_pdf(z)
            # log q(z)
            log_q = self.encoder.log_density(mu, log_var, z, log_det_j, z0)
            mc_samples.append(log_q - log_p)

        mc_tensor = torch.stack(mc_samples, dim=-1)
        mc_mean = torch.mean(mc_tensor, dim=-1)
        return mc_mean


# DOES NOT WORK
# class EncoderLinearLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, ratio=0.1, device=None):
#         # output dim is the dim of initial condition
#         # input dim is observation and action
#
#         super(EncoderLinearLSTM, self).__init__()
#
#         if device is None:
#             self.device = get_device()
#         else:
#             self.device = device
#
#         self.hidden_dim = hidden_dim
#         self.ratio = ratio
#
#         self.encoder_LSTM = EncoderLSTM(input_dim, hidden_dim, output_dim, False, device)
#
#         self.lin = nn.Linear(input_dim, output_dim).to(self.device)
#
#         self.log_var = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ELU(),
#             nn.Linear(hidden_dim, output_dim)
#         ).to(self.device)
#
#     def forward(self, x, a, mask):
#         # crude approximation using the first observation
#
#         y_in = torch.cat([x, a], dim=-1)[0]
#         mu1 = self.lin(y_in)
#         log_var1 = self.log_var(y_in)
#
#         if self.ratio > 0:
#             mu2, log_var2 = self.encoder_LSTM(x, a, mask)
#         else:
#             mu2, log_var2 = 0., 0.
#
#         mu = mu1 + self.ratio * mu2
#         log_var = log_var1 + self.ratio * log_var2
#
#         # scale mu
#         mu = torch.exp(mu) / 10
#         mask = torch.zeros_like(mu)
#         mask[:, 0] = 1
#         mu = mu * mask
#
#         # scale var
#         log_var = log_var - 5.
#
#         return mu, log_var
#
#     def reparameterize(self, mu, log_var):
#
#         return GaussianReparam.reparameterize(mu, log_var)
