from typing import NamedTuple


class RochConfig(NamedTuple):
    HillCure: float = 2
    HillPatho: float = 2
    ec50_patho: float = 1
    emax_patho: float = 1
    k_dexa: float = 1
    k_discure_immunereact: float = 1
    k_discure_immunity: float = 1
    k_disprog: float = 1
    k_immune_disease: float = 1
    k_immune_feedback: float = 1
    k_immune_off: float = 1
    k_immunity: float = 1
    # kel: float = 1/(5/24)
    kel: float = 1


class DataConfig(NamedTuple):
    n_sample: int = 1000
    obs_dim: int = 20
    # latent_dim: int = 12
    latent_dim: int = 6
    action_dim: int = 1
    t_max: int = 14
    step_size: int = 1
    sparsity: float = 0.5
    output_sparsity: float = 0.5
    output_sigma: float = 0.1  # 0.3
    dose_max: float = 1
    p_remove: float = 0.5


dim8_config = DataConfig(obs_dim=40, latent_dim=8, output_sparsity=1 - 0.375, output_sigma=0.2, dose_max=10)

dim12_config = DataConfig(obs_dim=80, latent_dim=12, output_sparsity=1 - 0.25, output_sigma=0.2, dose_max=10)


class ModelConfig(NamedTuple):
    encoder_latent_ratio: float = 2.0
    expert_only: bool = False
    neural_ode: bool = False
    path: str = "model/"


class OptimConfig(NamedTuple):
    lr: float = 0.01
    ode_method: str = "dopri5"
    niters: int = 400
    batch_size: int = 50
    test_freq: int = 10
    shuffle: bool = True
    n_restart: int = 5
    early_stop: int = 10


class EvalConfig(NamedTuple):
    t0: int = 5
