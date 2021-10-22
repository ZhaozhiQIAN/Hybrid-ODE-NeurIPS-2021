import argparse
import pickle

import dataloader
import model
import training_utils
import numpy as np
import torch
import torch.optim as optim
from sim_config import *


def run(seed: int,
        elbo: bool,
        device,
        eval_only,
        init_path,
        data_path,
        sample,
        data_config: DataConfig,
        roche_config: RochConfig,
        model_config: ModelConfig,
        optim_config: OptimConfig,
        eval_config: EvalConfig,
        encoder_output_dim=None,
        ablate=False,
        arg_itr=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda:' + str(device) if device != 'c' and torch.cuda.is_available() else 'cpu')

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
    test_freq = optim_config.test_freq
    early_stop = optim_config.early_stop

    # with open('data/datafile.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    with open(data_path, 'rb') as f:
        dg = pickle.load(f)

    # with open('data/datafile_high_dim.pkl', 'rb') as f:
    #     dg = pickle.load(f)

    dg.set_device(device)

    if not eval_only:
        dg.set_train_size(n_sample)

    print('Training with {} samples'.format(n_sample))
    # dg = dataloader.DataGeneratorRoche(n_sample, obs_dim, t_max, step_size,
    #                                    roche_config, output_sigma, latent_dim, sparsity, device=device)
    # dg.generate_data()
    # dg.split_sample()

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

    best_on_disk = 1e9

    for i in range(optim_config.n_restart):
        encoder = model.EncoderLSTM(obs_dim+action_dim, int(obs_dim*encoder_latent_ratio), encoder_output_dim, device=device, normalize=normalize)
        decoder = model.RocheExpertDecoder(obs_dim, encoder_output_dim, action_dim, t_max, step_size, roche=roche, method=ode_method, device=device, ablate=ablate)

        vi = model.VariationalInference(encoder, decoder, prior_log_pdf=prior, elbo=elbo)

        if eval_only:
            break

        if init_path is not None:
            checkpoint = torch.load(init_path + vi.model_name)
            vi.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            vi.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        params = list(vi.encoder.parameters()) + list(vi.decoder.output_function.parameters()) + list(vi.decoder.ode.ml_net.parameters())

        optimizer = optim.Adam(params, lr=lr)

        res = training_utils.variational_training_loop(niters=niters, data_generator=dg, model=vi,
                                                       batch_size=batch_size, optimizer=optimizer,
                                                       test_freq=test_freq, path=model_config.path,
                                                       best_on_disk=best_on_disk, early_stop=early_stop,
                                                       shuffle=optim_config.shuffle)
        vi, best_on_disk, training_time = res

    if eval_only:
        best_model = torch.load(path + vi.model_name)
        vi.encoder.load_state_dict(best_model['encoder_state_dict'])
        vi.decoder.load_state_dict(best_model['decoder_state_dict'])
        best_loss = best_model['best_loss']
        print('Overall best loss: {:.6f}'.format(best_loss))

    training_utils.evaluate(vi, dg, batch_size, eval_config.t0)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser('PKPD simulation')
    parser.add_argument('--method', choices=['expert', 'neural', 'hybrid'], default='False', type=str)
    parser.add_argument('--device', choices=['0', '1', 'c'], default='1', type=str)
    parser.add_argument('--seed', default=666, type=int)
    parser.add_argument('--sample', default=1000, type=int)
    parser.add_argument('--path', default=None, type=str)
    parser.add_argument('--restart', default=3, type=int)
    parser.add_argument('--arg_itr', default=None, type=int)
    parser.add_argument('--eval', default='n', type=str)
    parser.add_argument('--elbo', default='y', type=str)
    parser.add_argument('--init', default=None, type=str)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--t0', default=5, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--data_config', default=None, type=str)
    parser.add_argument('--encoder_output_dim', default=None, type=int)
    parser.add_argument('--data_path', default='data/datafile_dose_exp.pkl', type=str)
    parser.add_argument('--ablate', default=False, type=bool)

    args = parser.parse_args()
    method = args.method
    seed = args.seed
    device = args.device
    path = args.path
    sample = args.sample
    restart = args.restart
    eval_only = args.eval == 'y'
    init_path = args.init
    batch_size = args.batch_size
    data_path = args.data_path
    dc = args.data_config
    elbo = args.elbo == 'y'
    encoder_output_dim = args.encoder_output_dim
    arg_itr = args.arg_itr

    if dc == 'dim8':
        data_config = dim8_config
    elif dc == 'dim12':
        data_config = dim12_config
    else:
        data_config = DataConfig(n_sample=sample)
    roche_config = RochConfig()
    if method == 'expert':
        model_config = ModelConfig(expert_only=True, path=path)
    elif method == 'neural':
        model_config = ModelConfig(neural_ode=True, path=path)
    elif method == 'hybrid':
        model_config = ModelConfig(path=path)

    # todo: try no shuffle
    optim_config = OptimConfig(shuffle=False, n_restart=restart, batch_size=batch_size, lr=args.lr)
    eval_config = EvalConfig(t0=args.t0)
    run(seed, elbo, device, eval_only, init_path, data_path, sample, data_config, roche_config, model_config, optim_config, eval_config, encoder_output_dim, args.ablate, arg_itr)
