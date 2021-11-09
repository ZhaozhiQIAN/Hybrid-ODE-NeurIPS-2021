import pickle

import numpy as np
import torch

import dataloader
import sim_config


val_size=100
test_size=1000

data_config = sim_config.DataConfig(n_sample=1000+val_size+test_size)
n_sample = data_config.n_sample

action_dim = data_config.action_dim
t_max = data_config.t_max
step_size = data_config.step_size
sparsity = data_config.sparsity
p_remove = data_config.p_remove

output_sigma = 0.2
dose_max = 10

# on averaga three latents contribute to a measurement

# dim 12
output_sparsity = 1 - 0.25
latent_dim = 12
obs_dim = 80

roche_config = sim_config.RochConfig(kel=1)

seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cpu')

dg = dataloader.DataGeneratorRoche(n_sample, obs_dim, t_max, step_size,
                                   roche_config, output_sigma, dose_max, latent_dim, sparsity,
                                   p_remove=p_remove, output_sparsity=output_sparsity, device=device,
                                   val_size=val_size, test_size=test_size)
dg.generate_data()
dg.split_sample()

with open('data/datafile_dim12.pkl', 'wb') as f:
    pickle.dump(dg, f)