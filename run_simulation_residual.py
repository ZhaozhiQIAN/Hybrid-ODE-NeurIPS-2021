import argparse
import pickle

import numpy as np
import torch
import torch.optim as optim

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
    lr = optim_config.lr
    ode_method = optim_config.ode_method
    niters = optim_config.niters
    batch_size = min(optim_config.batch_size, n_sample)
    test_freq = optim_config.test_freq
    early_stop = optim_config.early_stop

    # with open('data/datafile.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    with open(data_path, 'rb') as f:
        dg = pickle.load(f)

    # with open('data/datafile_high_dim.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    dg.set_device(device)

    # dg.set_train_size(n_sample)

    print('Training with {} samples'.format(n_sample))

    # load models
    model_expert = init_and_load(data_config, optim_config, model_config_expert, dg, init_path=path)

    # calculate ensemble weight
    x = dg.data_val['measurements']
    a = dg.data_val['actions']
    mask = dg.data_val['masks']

    # learn ensemble weights
    with torch.no_grad():
        encoder_out = model_expert.encoder(x, a, mask)
        z0_hat = encoder_out[0]
        x_hat, _ = model_expert.decoder(z0_hat, a)

        residual = x - x_hat
        dg.data_val['measurements'] = residual.detach()

    n_sample = min(n_sample, residual.shape[1])
    dg.set_val_size(n_sample)
    print(dg.data_val['measurements'].shape)

    # start training
    best_on_disk = 1e9

    for i in range(optim_config.n_restart):
        vi = init_and_load(data_config, optim_config, model_config_ml, dg)

        if eval_only:
            break

        params = list(vi.encoder.parameters()) + list(vi.decoder.output_function.parameters()) + list(vi.decoder.ode.ml_net.parameters())

        optimizer = optim.Adam(params, lr=lr)

        res = training_utils.variational_training_loop(niters=niters, data_generator=dg, model=vi,
                                                       batch_size=batch_size, optimizer=optimizer,
                                                       test_freq=test_freq, path=model_config_ml.path,
                                                       best_on_disk=best_on_disk, early_stop=early_stop,
                                                       shuffle=optim_config.shuffle, train_fold='val')
        vi, best_on_disk, training_time = res

    print('Ensemble weights learned.')

    if eval_only:
        best_model = torch.load(model_config_ml.path + vi.model_name)
        vi.encoder.load_state_dict(best_model['encoder_state_dict'])
        vi.decoder.load_state_dict(best_model['decoder_state_dict'])
        best_loss = best_model['best_loss']
        print('Overall best loss: {:.6f}'.format(best_loss))
    print(model_config_ml.path + vi.model_name)

    if not horizon:
        training_utils.evaluate_ensemble(model_expert, vi, dg, batch_size, eval_config.t0)
    else:
        res = training_utils.evaluate_ensemble_horizon(model_expert, vi, dg, batch_size, eval_config.t0)

        with open(result_path, 'wb') as f:
            pickle.dump(res, f)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser('PKPD simulation')
    parser.add_argument('--method', choices=['residual'], default='residual', type=str)
    parser.add_argument('--device', choices=['0', '1', 'c'], default='1', type=str)
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--sample', default=1000, type=int)
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--res_path', default=None, type=str)
    parser.add_argument('--t0', default=5, type=int)
    parser.add_argument('--restart', default=3, type=int)
    parser.add_argument('--eval', default='n', type=str)
    parser.add_argument('--data_path', default='data/datafile_dose_exp.pkl', type=str)
    parser.add_argument('--data_config', default=None, type=str)
    parser.add_argument('--lr', default=0.01, type=float)
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

    if args.res_path is None:
        res_path = args.path + 'res'
    else:
        res_path = args.res_path

    if dc == 'dim8':
        data_config = sim_config.dim8_config
    elif dc == 'dim12':
        data_config = sim_config.dim12_config
    else:
        data_config = sim_config.DataConfig(n_sample=sample)
    roche_config = sim_config.RochConfig()

    model_config_expert = sim_config.ModelConfig(expert_only=True, path=path)
    model_config_ml = sim_config.ModelConfig(neural_ode=True, path=res_path)

    optim_config = sim_config.OptimConfig(shuffle=False, n_restart=restart, lr=args.lr)
    eval_config = sim_config.EvalConfig(t0=args.t0)

    run(seed, device, eval_only, data_path, sample, data_config,
        roche_config, model_config_expert, model_config_ml, optim_config, eval_config, args.horizon, args.result_path)

