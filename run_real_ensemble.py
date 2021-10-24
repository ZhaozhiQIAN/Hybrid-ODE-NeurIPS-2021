import dataloader
from sim_config import *
from scipy.optimize import nnls
import torch.optim as optim
import torch
import training_utils
import numpy as np
import model
import argparse
import pickle


def init_and_load(dg, encoder_output_dim, encoder_latent_ratio, ode_step_div, method, ode_method, init_path, t0=24):
    input_dim = dg.obs_dim + dg.action_dim + dg.static_dim + 1
    encoder = model.EncoderLSTMReal(input_dim, int(input_dim*encoder_latent_ratio), encoder_output_dim, output_all=False, reverse=False)

    obs_dim = dg.obs_dim
    action_dim = dg.action_dim
    static_dim = dg.static_dim
    hidden_dim = int((obs_dim + action_dim + static_dim) * encoder_latent_ratio)
    t_max = dg.t_max
    step_size = dg.step_size

    ode_step_size = dg.step_size / ode_step_div

    if method in ['hybrid', 'neural', '2nd', 'expert']:
        decoder = model.DecoderReal(obs_dim, encoder_output_dim, action_dim, static_dim,
                                    hidden_dim, t_max, step_size,
                                    method=ode_method, ode_step_size=ode_step_size, ode_type=method, t0=t0)
    else:
        decoder = model.DecoderRealBenchmark(obs_dim, encoder_output_dim, action_dim, static_dim,
                                             hidden_dim, t_max, step_size, ode_type=method, t0=t0)

    vi = model.VariationalInferenceReal(encoder, decoder, elbo=False, t0=t0, weight=weight)

    if init_path is not None:
        checkpoint = torch.load(init_path + vi.model_name)
        vi.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        vi.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    return vi


def run(seed=666,
        train_sample_size=None,
        method='ensemble',
        ode_method='explicit_adams',
        niters=1500,
        ode_step_div=1,
        encoder_output_dim=20,
        encoder_latent_ratio=1.2,
        init_path='model/'):

    np.random.seed(seed)
    torch.manual_seed(seed)

    roche_config = RochConfig()
    dg = dataloader.DataGeneratorReal(2097, 1, 1, 1, roche_config, 1, val_size=100, test_size=1000, latent_dim=10, data_type='5')
    dg.split_sample()
    if train_sample_size is not None:
        dg.set_train_size(train_sample_size)

    t0 = 24
    if train_sample_size is not None:
        init_path = init_path + str(train_sample_size)
    model_expert = init_and_load(dg, 4, encoder_latent_ratio, ode_step_div, 'expert', ode_method, init_path, t0)
    model_ml = init_and_load(dg, encoder_output_dim, encoder_latent_ratio, ode_step_div, 'gruode', ode_method, init_path, t0)

    # learn ensemble weight

    x = dg.data_val['measurements']
    a = dg.data_val['actions']
    mask = dg.data_val['masks']
    s = dg.data_val['statics']
    a_in = torch.cat([a, s], dim=-1)

    # learn ensemble weights
    encoder_out = model_expert.encoder(x, a_in, mask)
    z0_hat = encoder_out[0]
    x_hat, _ = model_expert.decoder(z0_hat, a_in, s)

    encoder_out_ml = model_ml.encoder(x, a_in, mask)
    z0_hat_ml = encoder_out_ml[0]
    x_hat_ml, _ = model_ml.decoder(z0_hat_ml, a, s)

    x = x[t0:, ...]
    weights_e = torch.zeros(x.shape[0], 1, x.shape[2]).to(x)
    weights_m = torch.zeros(x.shape[0], 1, x.shape[2]).to(x)

    for i in range(x.shape[0]):
        b = x[i, :, :].cpu().numpy().flatten()
        xe = x_hat[i, :, :].detach().cpu().numpy().flatten()
        xm = x_hat_ml[i, :, :].detach().cpu().numpy().flatten()
        A = np.stack([xe, xm], axis=1)
        w, _ = nnls(A, b)
        for j in range(x.shape[2]):
            weights_e[i, 0, j] = w[0]
            weights_m[i, 0, j] = w[1]
    print('Ensemble weights learned.')

    # evaluate
    data = dg.data_test

    x = data['measurements']
    a = data['actions']
    mask = data['masks']
    s = data['statics']

    with torch.no_grad():
        # Evaluate the goodness of point estimate
        a_in = torch.cat([a, s], dim=-1)
        encoder_out = model_expert.encoder(x[:t0], a_in[:t0], mask[:t0])
        z0_hat = encoder_out[0]
        x_hat1, _ = model_expert.decoder(z0_hat, a_in, data['statics'])

        encoder_out = model_ml.encoder(x[:t0], a_in[:t0], mask[:t0])
        z0_hat = encoder_out[0]
        x_hat2, _ = model_ml.decoder(z0_hat, data['actions'], data['statics'])

        x_hat = x_hat1 * weights_e + x_hat2 * weights_m

    eval_dict = {
        'x': x,
        'x_hat': x_hat,
        'mask': mask,
        'name': 'ensemble',
        'model_path': None
    }
    pickle.dump(eval_dict, open(init_path + 'ensemble' + 'eval.pkl', 'wb'))

    t1_list = [24 + 6, 24 + 12, 24 + 24, 24 + 24 * 3]


    for t1 in t1_list:
        a = torch.sum((x[t0:t1]-x_hat[:(t1-t0)]) ** 2 * mask[t0:t1], dim=(0, 2)) / torch.sum(mask[t0:t1], dim=(0, 2))
        a = a[~torch.isnan(a)]
        rmse = torch.sqrt(torch.mean(a))
        rmse_sd = training_utils.bootstrap_RMSE(a)

        print('rmse_x,{:.4f},{:.4f},{:.4f}'.format(t1, rmse, rmse_sd))


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser('Real data')
    parser.add_argument('--method', choices=['ensemble'], default='ensemble', type=str)
    parser.add_argument('--seed', default=666, type=int)

    parser.add_argument('--ode_method', default='explicit_adams', type=str)
    parser.add_argument('--ode_step_div', default=1, type=int)
    parser.add_argument('--encoder_output_dim', default=20, type=int)
    parser.add_argument('--encoder_latent_ratio', default=1.2, type=float)
    parser.add_argument('--weight', default='n', type=str)
    parser.add_argument('--niters', default=1500, type=int)
    parser.add_argument('--init_path', default='model/', type=str)
    parser.add_argument('--sample', default=None, type=int)

    args = parser.parse_args()

    weight = args.weight == 'y'        # return dmldt

    run(seed=args.seed,
        niters=args.niters,
        train_sample_size=args.sample,
        method=args.method,
        ode_method=args.ode_method,
        ode_step_div=args.ode_step_div,
        encoder_output_dim=args.encoder_output_dim,
        encoder_latent_ratio=args.encoder_latent_ratio,
        init_path=args.init_path)
