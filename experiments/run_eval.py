import argparse
import pickle

import numpy as np
import torch

import model
import sim_config
import training_utils


def run(
    seed: int,
    elbo: bool,
    device,
    eval_only,
    init_path,
    data_path,
    sample,
    data_config: sim_config.DataConfig,
    roche_config: sim_config.RochConfig,
    model_config: sim_config.ModelConfig,
    optim_config: sim_config.OptimConfig,
    eval_config: sim_config.EvalConfig,
    encoder_output_dim=None,
    ablate=False,
    arg_itr=None,
    result_path=None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda:" + str(device) if device != "c" and torch.cuda.is_available() else "cpu")

    # data config
    n_sample = sample
    obs_dim = data_config.obs_dim
    latent_dim = data_config.latent_dim
    action_dim = data_config.action_dim
    t_max = data_config.t_max
    step_size = data_config.step_size
    output_sigma = data_config.output_sigma
    sparsity = data_config.sparsity

    # optim config
    lr = optim_config.lr
    ode_method = optim_config.ode_method
    if arg_itr is None:
        niters = optim_config.niters
    else:
        niters = arg_itr
    batch_size = optim_config.batch_size

    with open(data_path, "rb") as f:
        dg = pickle.load(f)

    dg.set_device(device)

    if not eval_only:
        dg.set_train_size(n_sample)

    print("Training with {} samples".format(n_sample))

    # model config
    encoder_latent_ratio = model_config.encoder_latent_ratio
    if encoder_output_dim is None:
        if model_config.expert_only:
            encoder_output_dim = dg.expert_dim
        else:
            encoder_output_dim = dg.latent_dim

    if model_config.neural_ode:
        prior = None
        roche = False
        normalize = False
    else:
        prior = model.ExponentialPrior.log_density
        roche = True
        normalize = True

    encoder = model.EncoderLSTM(
        obs_dim + action_dim,
        int(obs_dim * encoder_latent_ratio),
        encoder_output_dim,
        device=device,
        normalize=normalize,
    )
    decoder = model.RocheExpertDecoder(
        obs_dim,
        encoder_output_dim,
        action_dim,
        t_max,
        step_size,
        roche=roche,
        method=ode_method,
        device=device,
        ablate=ablate,
    )

    vi = model.VariationalInference(encoder, decoder, prior_log_pdf=prior, elbo=elbo)

    best_model = torch.load(path + vi.model_name)
    vi.encoder.load_state_dict(best_model["encoder_state_dict"])
    vi.decoder.load_state_dict(best_model["decoder_state_dict"])
    best_loss = best_model["best_loss"]
    print("Overall best loss: {:.6f}".format(best_loss))

    res = training_utils.evaluate_horizon(vi, dg, batch_size, eval_config.t0)

    with open(result_path, "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser("PKPD simulation")
    parser.add_argument("--method", choices=["expert", "neural", "hybrid"], default="False", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="1", type=str)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument("--sample", default=1000, type=int)
    parser.add_argument("--path", default=None, type=str)
    parser.add_argument("--result_path", default=None, type=str)
    parser.add_argument("--restart", default=3, type=int)
    parser.add_argument("--arg_itr", default=None, type=int)
    parser.add_argument("--eval", default="n", type=str)
    parser.add_argument("--elbo", default="y", type=str)
    parser.add_argument("--init", default=None, type=str)
    parser.add_argument("--batch_size", default=50, type=int)
    parser.add_argument("--t0", default=5, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--data_config", default=None, type=str)
    parser.add_argument("--encoder_output_dim", default=None, type=int)
    parser.add_argument("--data_path", default="data/datafile_dose_exp.pkl", type=str)
    parser.add_argument("--ablate", default=False, type=bool)

    args = parser.parse_args()
    method = args.method
    seed = args.seed
    device = args.device
    path = args.path
    sample = args.sample
    restart = args.restart
    eval_only = args.eval == "y"
    init_path = args.init
    batch_size = args.batch_size
    data_path = args.data_path
    dc = args.data_config
    elbo = args.elbo == "y"
    encoder_output_dim = args.encoder_output_dim
    arg_itr = args.arg_itr

    assert eval_only

    if dc == "dim8":
        data_config = sim_config.dim8_config
    elif dc == "dim12":
        data_config = sim_config.dim12_config
    else:
        data_config = sim_config.DataConfig(n_sample=sample)
    roche_config = sim_config.RochConfig()
    if method == "expert":
        model_config = sim_config.ModelConfig(expert_only=True, path=path)
    elif method == "neural":
        model_config = sim_config.ModelConfig(neural_ode=True, path=path)
    elif method == "hybrid":
        model_config = sim_config.ModelConfig(path=path)

    optim_config = sim_config.OptimConfig(shuffle=False, n_restart=restart, batch_size=batch_size, lr=args.lr)
    eval_config = sim_config.EvalConfig(t0=args.t0)
    run(
        seed,
        elbo,
        device,
        eval_only,
        init_path,
        data_path,
        sample,
        data_config,
        roche_config,
        model_config,
        optim_config,
        eval_config,
        encoder_output_dim,
        args.ablate,
        arg_itr,
        args.result_path,
    )
