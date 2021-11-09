import pickle

import numpy as np
import scipy.integrate
import torch

from global_config import DTYPE, get_device


class DataGeneratorRoche:

    def __init__(self, n_sample, obs_dim, t_max, step_size, roche_config, output_sigma, dose_max=0, latent_dim=4,
                 sparsity=0.5, output_sparsity=0., val_size=100, test_size=200, p_remove=0, device=None, dtype=DTYPE):
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.dtype = dtype

        self.n_sample = n_sample
        self.obs_dim = obs_dim
        self.latent_dim = int(latent_dim)
        self.expert_dim = int(4)
        self.ml_dim = self.latent_dim - self.expert_dim
        self.sparsity = sparsity
        self.action_dim = int(1)
        self.expanded = True if self.ml_dim > 0 else False
        self.t_max = t_max
        self.step_size = step_size
        self.time_dim = int(t_max / step_size + 1)
        self.roche_config = roche_config
        self.dose_max = dose_max
        self.p_remove = p_remove
        self.output_sparsity = output_sparsity

        self.output_coef = np.random.randn(obs_dim, self.latent_dim + self.action_dim) * \
                           np.random.binomial(1, 1 - self.output_sparsity, (obs_dim, self.latent_dim + self.action_dim))
        self.output_sigma = output_sigma
        self.ml_coef = np.random.randn(self.latent_dim, self.ml_dim) * \
                       np.random.binomial(1, 1 - self.sparsity, (self.latent_dim, self.ml_dim)) / self.latent_dim

        self.val_size = int(val_size)
        self.test_size = int(test_size)
        self.train_size = int(n_sample - val_size - test_size)

        self.measurements = None
        self.actions = None
        self.latents = None
        self.masks = None
        self.data_train, self.data_val, self.data_test = None, None, None

    def set_device(self, device):
        self.device = device
        self.measurements = self.measurements.to(device)
        self.actions = self.actions.to(device)
        self.latents = self.latents.to(device)
        self.masks = self.masks.to(device)

        for a in [self.data_train, self.data_val, self.data_test]:
            for k in ['measurements', 'actions', 'latents', 'masks']:
                a[k] = a[k].to(device)

    def set_train_size(self, n_sample):
        train_sample_size = n_sample - self.val_size - self.test_size
        self.train_size = train_sample_size
        self.n_sample = n_sample
        print('train_size', self.train_size)
        print('n_sample', self.n_sample)
        for k in ['measurements', 'actions', 'latents', 'masks']:
            self.data_train[k] = self.data_train[k][:, :train_sample_size, :]

    def set_val_size(self, n_val):
        self.val_size = n_val
        for k in ['measurements', 'actions', 'latents', 'masks']:
            self.data_val[k] = self.data_val[k][:, :n_val, :]

    def solve(self, init, dose_times, dose_amount):

        def dose_at_time(t):
            return dose_amount * np.sum(
                np.exp(self.roche_config.kel * (dose_times - t) * (t >= dose_times)) * (t >= dose_times))

        def dose_at_time_discrete(t):
            return dose_amount * np.max((dose_times - t) == 0)

        def ode_roche(t, y, HillCure,
                      HillPatho,
                      ec50_patho,
                      emax_patho,
                      k_dexa,
                      k_discure_immunereact,  # k_innateimmreact
                      k_discure_immunity,  # k_dis_cure
                      k_disprog,
                      k_immune_disease,  # k_init_react
                      k_immune_feedback,  # k_pos_feedb
                      k_immune_off,  # k_out
                      k_immunity,
                      kel,
                      ):
            Disease = y[0]
            ImmuneReact = y[1]
            Immunity = y[2]
            Dose2 = y[3]

            dxdt1 = Disease * k_disprog - Disease * Immunity ** HillCure * k_discure_immunity - Disease * ImmuneReact * k_discure_immunereact

            Dose = dose_at_time(t)

            dxdt2 = Disease * k_immune_disease - ImmuneReact * k_immune_off + Disease * ImmuneReact * k_immune_feedback + (
                    ImmuneReact ** HillPatho * emax_patho) / (
                            ec50_patho ** HillPatho + ImmuneReact ** HillPatho) - Dose2 * ImmuneReact * k_dexa
            dxdt3 = ImmuneReact * k_immunity
            dxdt4 = kel * Dose - kel * Dose2

            if self.expanded:
                ml_states = np.tanh(np.matmul(y, self.ml_coef))
                return [dxdt1, dxdt2, dxdt3, dxdt4] + list(ml_states)
            else:
                return [dxdt1, dxdt2, dxdt3, dxdt4]

        ode = scipy.integrate.ode(ode_roche).set_integrator('lsoda')
        ode.set_initial_value(init, 0).set_f_params(*self.roche_config)

        t1 = self.t_max
        dt = self.step_size

        res_list = [init]

        while ode.successful() and ode.t < t1:
            res = ode.integrate(ode.t + dt, ode.t + dt)
            res_list.append(res)

        # latents
        latents = np.stack(res_list, axis=-1)

        # calculate outputs
        # todo: simulate measurements with more complex model
        # D, T
        input_factor = np.concatenate((latents, np.ones((1, latents.shape[1]))), axis=0)
        output = np.matmul(self.output_coef, input_factor)
        output = output + np.random.randn(output.shape[0], output.shape[1]) * self.output_sigma
        measurements = output
        # measurements = np.tanh(output)
        # measurements = latents

        action_list = []
        for time in np.arange(0, t1 + dt, dt):
            action_list.append(dose_at_time_discrete(time))

        actions = np.array(action_list)[None, :]

        if self.time_dim != latents.shape[1]:
            fill_in = np.zeros((self.latent_dim, self.time_dim - latents.shape[1]))
            latents = np.concatenate((latents, fill_in), axis=1)

            fill_in = np.zeros((self.obs_dim, self.time_dim - measurements.shape[1]))
            measurements = np.concatenate((measurements, fill_in), axis=1)

            fill_in = np.zeros((self.action_dim, self.time_dim - actions.shape[1]))
            actions = np.concatenate((actions, fill_in), axis=1)

        mask = np.ones(self.time_dim)
        mask[latents.shape[1]:] = 0
        mask = mask[None, :]
        # measurements: D, T
        # actions: D, T
        # latents: D, T
        return measurements, actions, latents, mask

    def get_initial_conditions(self):
        # B, D
        init = np.random.exponential(scale=0.01, size=(self.n_sample, self.latent_dim))
        # init = np.random.randn(self.n_sample, self.latent_dim)
        # init[:, 1:] = 0.
        return init

    def get_action(self):
        # dose time
        dose_list = []
        for i in range(self.n_sample):
            dose = np.random.choice(self.t_max, size=1, replace=False)
            dose_list.append(dose)

        # N, N_dose
        dose_time = np.sort(np.stack(dose_list, axis=0))

        # todo: set non-zero dose amount slows down training
        dose_amount = np.random.rand(self.n_sample) * self.dose_max

        # dose_time: N, N_dose
        # dose amount: N
        return dose_time, dose_amount

    def _make_tensor(self, x):
        if type(x) == np.ndarray:
            return torch.tensor(x, dtype=self.dtype, device=self.device)
        else:
            return x.to(dtype=self.dtype, device=self.device)

    def generate_data(self):

        # simulate initial conditions
        init = self.get_initial_conditions()

        # simulate actions
        dose_time, dose_amount = self.get_action()

        self.dose_time = dose_time
        self.dose_amount = dose_amount

        # simulate measurements and latents
        m_list = []
        a_list = []
        l_list = []
        mask_list = []

        for i in range(self.n_sample):
            measurements, actions, latents, masks = self.solve(init[i], dose_time[i], dose_amount[i])
            m_list.append(measurements)
            a_list.append(actions)
            l_list.append(latents)
            mask_list.append(masks)

        # return m_list, a_list, l_list
        measurements = self._make_tensor(np.stack(m_list, axis=0).transpose((2, 0, 1)))
        self.actions = self._make_tensor(np.stack(a_list, axis=0).transpose((2, 0, 1)))
        self.latents = self._make_tensor(np.stack(l_list, axis=0).transpose((2, 0, 1)))
        masks = self._make_tensor(np.stack(mask_list, axis=0).transpose((2, 0, 1)))

        # normalize the measurements
        self.measurements = (measurements - torch.mean(measurements, dim=(0, 1))) / torch.std(measurements, dim=(0, 1))

        # create irregular samples
        selected = (torch.rand_like(measurements) > self.p_remove) * 1.

        self.masks = masks * selected

        assert self.measurements.shape == (self.time_dim, self.n_sample, self.obs_dim)
        assert self.actions.shape == (self.time_dim, self.n_sample, self.action_dim)
        assert self.latents.shape == (self.time_dim, self.n_sample, self.latent_dim)

    def split_sample(self):

        data_train = {
            'measurements': self.measurements[:, :self.train_size, :],
            'actions': self.actions[:, :self.train_size, :],
            'latents': self.latents[:, :self.train_size, :],
            'masks': self.masks[:, :self.train_size, :],
        }

        data_val = {
            'measurements': self.measurements[:, self.train_size:(self.train_size + self.val_size), :],
            'actions': self.actions[:, self.train_size:(self.train_size + self.val_size), :],
            'latents': self.latents[:, self.train_size:(self.train_size + self.val_size), :],
            'masks': self.masks[:, self.train_size:(self.train_size + self.val_size), :],
        }

        data_test = {
            'measurements': self.measurements[:, (self.train_size + self.val_size):, :],
            'actions': self.actions[:, (self.train_size + self.val_size):, :],
            'latents': self.latents[:, (self.train_size + self.val_size):, :],
            'masks': self.masks[:, (self.train_size + self.val_size):, :],
        }

        self.data_train, self.data_val, self.data_test = data_train, data_val, data_test

    def _get_index_random(self, N, k):
        indices = self._make_tensor(np.random.choice(N, k, replace=False)).to(torch.int64)
        return indices

    def get_mini_batch(self, fold, batch_size):
        assert fold in ('train', 'val', 'test')

        if fold == 'train':
            data = self.data_train
        elif fold == 'val':
            data = self.data_val
        else:
            data = self.data_test

        n_sample = data['measurements'].shape[1]

        indices = self._get_index_random(n_sample, batch_size)
        data_batch = {
            'measurements': data['measurements'][:, indices, :],
            'actions': data['actions'][:, indices, :],
            'latents': data['latents'][:, indices, :],
            'masks': data['masks'][:, indices, :],
        }
        return data_batch

    def get_split(self, fold, batch_size, chunk=0):
        assert fold in ('train', 'val', 'test')

        if fold == 'train':
            data = self.data_train
        elif fold == 'val':
            data = self.data_val
        else:
            data = self.data_test

        index_begin = chunk * batch_size
        index_end = (chunk + 1) * batch_size

        data_batch = {
            'measurements': data['measurements'][:, index_begin:index_end, :],
            'actions': data['actions'][:, index_begin:index_end, :],
            'latents': data['latents'][:, index_begin:index_end, :],
            'masks': data['masks'][:, index_begin:index_end, :],
        }
        return data_batch


class DataGeneratorReal(DataGeneratorRoche):

    def __init__(self, n_sample, obs_dim, t_max, step_size, roche_config, output_sigma, dose_max=0, latent_dim=4,
                 sparsity=0.5, output_sparsity=0., val_size=100, test_size=200, p_remove=0, device=None, dtype=DTYPE, data_type='', data_path='../data/'):
        super().__init__(n_sample, obs_dim, t_max, step_size, roche_config, output_sigma, dose_max, latent_dim,
                 sparsity, output_sparsity, val_size, test_size, p_remove, device, dtype)

        masks = pickle.load(open(data_path + "array_xt_mask{}.pkl".format(data_type), "rb"))

        self.n_sample = masks.shape[1]
        self.obs_dim = masks.shape[2]
        self.t_max = masks.shape[0]
        self.step_size = 1.
        self.time_dim = masks.shape[0]

        # load data
        self.statics = self._make_tensor(pickle.load(open(data_path + "array_x_constant.pkl", "rb")))[None, :, :]
        self.statics = torch.cat([self.statics] * self.time_dim, dim=0)
        self.masks = self._make_tensor(pickle.load(open(data_path + "array_xt_mask{}.pkl".format(data_type), "rb")))
        self.measurements = self._make_tensor(pickle.load(open(data_path + "array_xt{}.pkl".format(data_type), "rb")))
        self.actions = self._make_tensor(pickle.load(open(data_path + "array_at{}.pkl".format(data_type), "rb")))
        self.latents = torch.zeros_like(self.masks)[:, :, :self.latent_dim]

        self.static_dim = self.statics.shape[2]

        # print(self.measurements.shape)
        # print(self.actions.shape)
        # print(self.latents.shape)
        # print(self.time_dim)
        # print(self.n_sample)
        # print(self.obs_dim)
        # print(self.action_dim)
        # print(self.latent_dim)
        assert self.measurements.shape == (self.time_dim, self.n_sample, self.obs_dim)
        assert self.actions.shape == (self.time_dim, self.n_sample, self.action_dim)
        assert self.latents.shape == (self.time_dim, self.n_sample, self.latent_dim)

    def split_sample(self):

        data_train = {
            'measurements': self.measurements[:, :self.train_size, :],
            'actions': self.actions[:, :self.train_size, :],
            'latents': self.latents[:, :self.train_size, :],
            'masks': self.masks[:, :self.train_size, :],
            'statics': self.statics[:, :self.train_size, :],
        }

        data_val = {
            'measurements': self.measurements[:, self.train_size:(self.train_size + self.val_size), :],
            'actions': self.actions[:, self.train_size:(self.train_size + self.val_size), :],
            'latents': self.latents[:, self.train_size:(self.train_size + self.val_size), :],
            'masks': self.masks[:, self.train_size:(self.train_size + self.val_size), :],
            'statics': self.statics[:, self.train_size:(self.train_size + self.val_size), :],
        }

        data_test = {
            'measurements': self.measurements[:, (self.train_size + self.val_size):, :],
            'actions': self.actions[:, (self.train_size + self.val_size):, :],
            'latents': self.latents[:, (self.train_size + self.val_size):, :],
            'masks': self.masks[:, (self.train_size + self.val_size):, :],
            'statics': self.statics[:, (self.train_size + self.val_size):, :],
        }

        self.data_train, self.data_val, self.data_test = data_train, data_val, data_test

    def get_mini_batch(self, fold, batch_size):
        assert fold in ('train', 'val', 'test')

        if fold == 'train':
            data = self.data_train
        elif fold == 'val':
            data = self.data_val
        else:
            data = self.data_test

        n_sample = data['measurements'].shape[1]

        indices = self._get_index_random(n_sample, batch_size)
        data_batch = {
            'measurements': data['measurements'][:, indices, :],
            'actions': data['actions'][:, indices, :],
            'latents': data['latents'][:, indices, :],
            'masks': data['masks'][:, indices, :],
            'statics': data['statics'][:, indices, :],
        }
        return data_batch

    def set_train_size(self, train_sample_size):
        self.train_size = train_sample_size
        self.n_sample = train_sample_size + self.val_size + self.test_size
        print('train_size', self.train_size)
        print('n_sample', self.n_sample)
        for k in ['measurements', 'actions', 'latents', 'masks', 'statics']:
            self.data_train[k] = self.data_train[k][:, :train_sample_size, :]


    def get_split(self, fold, batch_size, chunk=0):
        assert fold in ('train', 'val', 'test')

        if fold == 'train':
            data = self.data_train
        elif fold == 'val':
            data = self.data_val
        else:
            data = self.data_test

        index_begin = chunk * batch_size
        index_end = (chunk + 1) * batch_size

        data_batch = {
            'measurements': data['measurements'][:, index_begin:index_end, :],
            'actions': data['actions'][:, index_begin:index_end, :],
            'latents': data['latents'][:, index_begin:index_end, :],
            'masks': data['masks'][:, index_begin:index_end, :],
            'statics': data['statics'][:, index_begin:index_end, :],
        }
        return data_batch