import argparse
import pickle

import numpy as np
import torch
from scipy.optimize import nnls

import model
import sim_config
import training_utils


def init_and_load(data_config, optim_config, model_config, dg, init_path=None):
    obs_dim = data_config.obs_dim
    action_dim = data_config.action_dim
    t_max = data_config.t_max
    step_size = data_config.step_size
    ode_method = optim_config.ode_method


    # model config
    encoder_latent_ratio = model_config.encoder_latent_ratio
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


    encoder = model.EncoderLSTM(obs_dim+action_dim, int(obs_dim*encoder_latent_ratio), encoder_output_dim, device=dg.device, normalize=normalize)
    decoder = model.RocheExpertDecoder(obs_dim, encoder_output_dim, action_dim, t_max, step_size, roche=roche, method=ode_method, device=dg.device)

    vi = model.VariationalInference(encoder, decoder, prior_log_pdf=prior, elbo=True)

    if init_path is not None:
        checkpoint = torch.load(init_path + vi.model_name)
        vi.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        vi.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return vi


def run(seed: int,
        device,
        eval_only,
        data_path,
        sample,
        data_config: sim_config.DataConfig,
        roche_config: sim_config.RochConfig,
        model_config_expert: sim_config.ModelConfig,
        model_config_ml: sim_config.ModelConfig,
        optim_config: sim_config.OptimConfig,
        eval_config: sim_config.EvalConfig,
        horizon=False,
        result_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:' + str(device) if device != 'c' and torch.cuda.is_available() else 'cpu')

    # data config
    n_sample = sample

    # optim config
    batch_size = optim_config.batch_size

    # with open('data/datafile.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    with open(data_path, 'rb') as f:
        dg = pickle.load(f)

    # with open('data/datafile_high_dim.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    dg.set_device(device)

    # dg.set_train_size(n_sample)

    # print('Training with {} samples'.format(n_sample))

    # load models
    model_expert = init_and_load(data_config, optim_config, model_config_expert, dg, init_path=path)
    model_ml = init_and_load(data_config, optim_config, model_config_ml, dg, init_path=path)

    # calculate ensemble weight
    # size = min(dg.train_size, dg.val_size)
    x = dg.data_val['measurements'][:, :n_sample, :]
    a = dg.data_val['actions'][:, :n_sample, :]
    mask = dg.data_val['masks'][:, :n_sample, :]

    print(a.shape)

    # learn ensemble weights
    encoder_out = model_expert.encoder(x, a, mask)
    z0_hat = encoder_out[0]
    x_hat, _ = model_expert.decoder(z0_hat, a)

    encoder_out_ml = model_ml.encoder(x, a, mask)
    z0_hat_ml = encoder_out_ml[0]
    x_hat_ml, _ = model_ml.decoder(z0_hat_ml, a)

    weights_e = torch.zeros(x.shape[0], 1, x.shape[2]).to(x)
    weights_m = torch.zeros(x.shape[0], 1, x.shape[2]).to(x)
    # for i in range(eval_config.t0, x.shape[0]):
    #     for j in range(x.shape[2]):
    #         b = x[i, :, j].cpu().numpy()
    #         xe = x_hat[i, :, j].detach().cpu().numpy()
    #         xm = x_hat_ml[i, :, j].detach().cpu().numpy()
    #         A = np.stack([xe, xm], axis=1)
    #         w, _ = nnls(A, b)
    #         weights_e[i, 0, j] = w[0]
    #         weights_m[i, 0, j] = w[1]

    for i in range(eval_config.t0, x.shape[0]):
        b = x[i, :, :].cpu().numpy().flatten()
        xe = x_hat[i, :, :].detach().cpu().numpy().flatten()
        xm = x_hat_ml[i, :, :].detach().cpu().numpy().flatten()
        A = np.stack([xe, xm], axis=1)
        w, _ = nnls(A, b)
        for j in range(x.shape[2]):
            weights_e[i, 0, j] = w[0]
            weights_m[i, 0, j] = w[1]
    print('Ensemble weights learned.')
    if not horizon:
        training_utils.evaluate_ensemble(model_expert, model_ml, dg, batch_size, eval_config.t0,
                                         weight_expert=weights_e, weight_ml=weights_m)
    else:
        res = training_utils.evaluate_ensemble_horizon(model_expert, model_ml, dg, batch_size, eval_config.t0,
                                         weight_expert=weights_e, weight_ml=weights_m)

        with open(result_path, 'wb') as f:
            pickle.dump(res, f)




if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser('PKPD simulation')
    parser.add_argument('--method', choices=['ensemble'], default='ensemble', type=str)
    parser.add_argument('--device', choices=['0', '1', 'c'], default='1', type=str)
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--sample', default=1000, type=int)
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--t0', default=5, type=int)
    parser.add_argument('--restart', default=5, type=int)
    parser.add_argument('--eval', default='n', type=str)
    parser.add_argument('--data_path', default='data/datafile_dose_exp.pkl', type=str)
    parser.add_argument('--data_config', default=None, type=str)
    parser.add_argument('--horizon', default=False, type=bool)
    parser.add_argument('--result_path', default=None, type=str)

    args = parser.parse_args()
    method = args.method
    seed = args.seed
    device = args.device
    path = args.path
    sample = args.sample
    restart = args.restart
    eval_only = args.eval == 'y'
    data_path = args.data_path
    dc = args.data_config

    if dc == 'dim8':
        data_config = sim_config.dim8_config
    elif dc == 'dim12':
        data_config = sim_config.dim12_config
    else:
        data_config = sim_config.DataConfig(n_sample=sample)
    roche_config = sim_config.RochConfig()

    model_config_expert = sim_config.ModelConfig(expert_only=True, path=path)
    model_config_ml = sim_config.ModelConfig(neural_ode=True, path=path)

    optim_config = sim_config.OptimConfig(shuffle=False, n_restart=restart)
    eval_config = sim_config.EvalConfig(t0=args.t0)

    run(seed, device, eval_only, data_path, sample, data_config,
        roche_config, model_config_expert, model_config_ml, optim_config, eval_config, args.horizon, args.result_path)

