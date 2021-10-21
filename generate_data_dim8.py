import pickle

import numpy as np
import torch

import dataloader
from sim_config import *


val_size=100
test_size=1000

data_config = DataConfig(n_sample=1000+val_size+test_size)
n_sample = data_config.n_sample

action_dim = data_config.action_dim
t_max = data_config.t_max
step_size = data_config.step_size
sparsity = data_config.sparsity
p_remove = data_config.p_remove

output_sigma = 0.2
dose_max = 10

# on averaga three latents contribute to a measurement

# dim 8
output_sparsity = 1 - 0.375
latent_dim = 8
obs_dim = 40
roche_config = RochConfig(kel=1)

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


with open('data/datafile_dim8.pkl', 'wb') as f:
    pickle.dump(dg, f)
