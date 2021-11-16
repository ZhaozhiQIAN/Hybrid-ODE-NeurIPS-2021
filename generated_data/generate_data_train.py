import pickle

import numpy as np
import torch

import dataloader
import sim_config

data_config = sim_config.DataConfig(n_sample=1300)
n_sample = data_config.n_sample
obs_dim = data_config.obs_dim
latent_dim = data_config.latent_dim
action_dim = data_config.action_dim
t_max = data_config.t_max
step_size = data_config.step_size

p_remove = data_config.p_remove

output_sigma = 0.2

sparsity = data_config.sparsity
output_sparsity = 0.5
dose_max = 10

roche_config = sim_config.RochConfig(kel=1)

seed = 666
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")

dg = dataloader.DataGeneratorRoche(
    n_sample,
    obs_dim,
    t_max,
    step_size,
    roche_config,
    output_sigma,
    dose_max,
    latent_dim,
    sparsity,
    p_remove=p_remove,
    output_sparsity=output_sparsity,
    device=device,
)
dg.generate_data()
dg.split_sample()

with open("data/datafile_dose_exp.pkl", "wb") as f:
    pickle.dump(dg, f)
