import torch
import time
import numpy as np
import properscoring as ps


def variational_training_loop(niters, data_generator, model, batch_size, optimizer, test_freq, best_on_disk=1e9, early_stop=5, path='model/', shuffle=True, train_fold='train'):

    best_loss = 1e9

    early_stop_counter = 0
    if train_fold == 'train':
        train_chunk = data_generator.train_size // batch_size
    else:
        train_chunk = data_generator.val_size // batch_size

    start = time.time()
    for itr in range(1, niters + 1):
        # get mini-batch of training data
        if shuffle:
            data = data_generator.get_mini_batch(train_fold, batch_size)
        else:
            data = data_generator.get_split(train_fold, batch_size, itr % train_chunk)

        optimizer.zero_grad()

        # print('compute loss')
        # loss = model.loss(data)
        #
        try:
            loss = model.loss(data)
        except RuntimeError as e:
            print(e)
            break
        # print(loss.item())

        loss.backward()
        # print('calculate backward')

        optimizer.step()
        # print('update params')

        if itr % test_freq == 0:
            with torch.no_grad():

                total_loss = 0
                for chunk in range(data_generator.val_size // batch_size):
                    data = data_generator.get_split('val', batch_size, chunk)
                    try:
                        total_loss += model.loss(data).item()
                    except RuntimeError as e:
                        total_loss += 1e9
                        print(e)
                        break
                print('Iter {:04d} | Total Loss {:.6f} | Train Loss {:.6f}'.format(itr, total_loss, loss.item()))
                if total_loss < best_loss:
                    best_loss = total_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if total_loss < best_on_disk:
                    best_on_disk = total_loss
                    model.save(path, itr, best_on_disk)

        if early_stop_counter >= early_stop:
            break

    end = time.time()

    # load best model
    try:
        best_model = torch.load(path + model.model_name)
    except FileNotFoundError:
        model.save(path, 0, best_on_disk)
        best_model = torch.load(path + model.model_name)

    model.encoder.load_state_dict(best_model['encoder_state_dict'])
    model.decoder.load_state_dict(best_model['decoder_state_dict'])
    best_loss = best_model['best_loss']
    print('Time: {}'.format(end-start))
    print('Overall best loss: {:.6f}'.format(best_loss))

    return model, best_loss, end - start


def evaluate(model, data_generator, batch_size, t0, mc_itr=50, real=False):
    with torch.no_grad():
        # sample-level rmse
        total_rmse_z0 = list()
        total_rmse_x = list()
        total_cprs_z0 = list()
        total_cprs_x = list()

        for chunk in range(data_generator.test_size // batch_size):
            data = data_generator.get_split('test', batch_size, chunk)

            x = data['measurements'][:t0]
            a = data['actions'][:t0]
            mask = data['masks'][:t0]
            z0 = data['latents'][0]
            if real:
                s = data['statics'][:t0]

            # Evaluate the goodness of point estimate
            if real:
                a_in = torch.cat([a, s], dim=-1)
                encoder_out = model.encoder(x, a_in, mask)
                z0_hat = encoder_out[0]
                x_hat, _ = model.decoder(z0_hat, data['actions'], data['statics'])
            else:
                encoder_out = model.encoder(x, a, mask)
                z0_hat = encoder_out[0]
                x_hat, _ = model.decoder(z0_hat, data['actions'])

            x_hat = x_hat[t0:, ...]
            total_rmse_z0.append(torch.sum((z0[:, :data_generator.expert_dim] - z0_hat[:, :data_generator.expert_dim])**2, dim=1))

            # predicting future
            x_test = data['measurements'][t0:]
            mask_test = data['masks'][t0:]
            total_rmse_x.append(torch.sum((x_test - x_hat) ** 2 * mask_test, dim=(0, 2)) / torch.sum(mask_test, dim=(0, 2)))

            z_list = list()
            x_hat_list = list()

            for i in range(mc_itr):
                z_ = model.encoder.reparameterize(*encoder_out)
                if real:
                    x_hat, _ = model.decoder(z_, data['actions'], data['statics'])
                else:
                    x_hat, _ = model.decoder(z_, data['actions'])
                z_list.append(z_)
                x_hat_list.append(x_hat)

            # B, D, MC
            z_mat = torch.stack(z_list, dim=-1)

            # B, D
            z_cprs = np.zeros(z0[:, :data_generator.expert_dim].shape)
            for d1 in range(z0.shape[0]):
                for d2 in range(data_generator.expert_dim):
                    truth = z0[d1, d2].item()
                    pred = z_mat[d1, d2, :].cpu().numpy()
                    z_cprs[d1, d2] = ps.crps_ensemble(truth, pred)
            # B
            z_cprs = np.mean(z_cprs, axis=1)
            total_cprs_z0.append(z_cprs)

            # T0, B, D, MC
            x_hat_mat = torch.stack(x_hat_list, dim=-1)[t0:, ...]
            x_cprs = np.zeros(x_test.shape)
            for d1 in range(x_test.shape[0]):
                for d2 in range(x_test.shape[1]):
                    for d3 in range(x_test.shape[2]):
                        truth = x_test[d1, d2, d3].item()
                        pred = x_hat_mat[d1, d2, d3, :].cpu().numpy()
                        x_cprs[d1, d2, d3] = ps.crps_ensemble(truth, pred)
            x_cprs = np.mean(x_cprs, axis=(0, 2))
            total_cprs_x.append(x_cprs)

        total_rmse_z0 = torch.cat(total_rmse_z0)
        rmse_z0 = torch.sqrt(torch.mean(total_rmse_z0)).item()
        rmse_z0_sd = bootstrap_RMSE(total_rmse_z0)

        total_cprs_z0 = np.concatenate(total_cprs_z0)
        cprs_z0 = np.mean(total_cprs_z0)
        cprs_z0_sd = np.std(total_cprs_z0) / np.sqrt(len(total_cprs_z0))

        total_rmse_x = torch.cat(total_rmse_x)
        total_rmse_x = total_rmse_x[~torch.isnan(total_rmse_x)]
        rmse_x = torch.sqrt(torch.mean(total_rmse_x)).item()
        rmse_x_sd = bootstrap_RMSE(total_rmse_x)

        total_cprs_x = np.concatenate(total_cprs_x)
        cprs_x = np.mean(total_cprs_x)
        cprs_x_sd = np.std(total_cprs_x) / np.sqrt(len(total_cprs_x))


        print('rmse_z0,{:.4f},{:.4f}'.format(rmse_z0, rmse_z0_sd))
        print('rmse_x,{:.4f},{:.4f}'.format(rmse_x, rmse_x_sd))
        print('cprs_z0,{:.4f},{:.4f}'.format(cprs_z0, cprs_z0_sd))
        print('cprs_x,{:.4f},{:.4f}'.format(cprs_x, cprs_x_sd))

        return rmse_z0, rmse_z0_sd, cprs_z0, rmse_x, rmse_x_sd, cprs_x


def evaluate_horizon(model, data_generator, batch_size, t0, mc_itr=10, real=False):
    with torch.no_grad():
        # sample-level rmse
        total_rmse_x = list()
        total_cprs_x = list()

        for chunk in range(data_generator.test_size // batch_size):
            data = data_generator.get_split('test', batch_size, chunk)

            x = data['measurements'][:t0]
            a = data['actions'][:t0]
            mask = data['masks'][:t0]
            z0 = data['latents'][0]
            if real:
                s = data['statics'][:t0]

            # Evaluate the goodness of point estimate
            if real:
                a_in = torch.cat([a, s], dim=-1)
                encoder_out = model.encoder(x, a_in, mask)
                z0_hat = encoder_out[0]
                x_hat, _ = model.decoder(z0_hat, data['actions'], data['statics'])
            else:
                encoder_out = model.encoder(x, a, mask)
                z0_hat = encoder_out[0]
                x_hat, _ = model.decoder(z0_hat, data['actions'])

            x_hat = x_hat[t0:, ...]

            # predicting future
            x_test = data['measurements'][t0:]
            mask_test = data['masks'][t0:]
            # T, B
            total_rmse_x.append(torch.sum((x_test - x_hat) ** 2 * mask_test, dim=2) / torch.sum(mask_test, dim=2))

            z_list = list()
            x_hat_list = list()

            for i in range(mc_itr):
                z_ = model.encoder.reparameterize(*encoder_out)
                if real:
                    x_hat, _ = model.decoder(z_, data['actions'], data['statics'])
                else:
                    x_hat, _ = model.decoder(z_, data['actions'])
                z_list.append(z_)
                x_hat_list.append(x_hat)

            # T0, B, D, MC
            x_hat_mat = torch.stack(x_hat_list, dim=-1)[t0:, ...]
            x_cprs = np.zeros(x_test.shape)
            for d1 in range(x_test.shape[0]):
                for d2 in range(x_test.shape[1]):
                    for d3 in range(x_test.shape[2]):
                        truth = x_test[d1, d2, d3].item()
                        pred = x_hat_mat[d1, d2, d3, :].cpu().numpy()
                        x_cprs[d1, d2, d3] = ps.crps_ensemble(truth, pred)
            x_cprs = np.mean(x_cprs, axis=2)
            total_cprs_x.append(x_cprs)

        # T, B
        total_rmse_x = torch.cat(total_rmse_x, dim=1)
        # T
        rmse_x = torch.sqrt(torch.nanmean(total_rmse_x, dim=1)).numpy()
        rmse_x_sd_list = []

        for i in range(rmse_x.shape[0]):
            rmse_x_sd_list.append(bootstrap_RMSE(total_rmse_x[i]))

        rmse_x_sd = np.array(rmse_x_sd_list)

        total_cprs_x = np.concatenate(total_cprs_x, axis=1)
        cprs_x = np.mean(total_cprs_x, axis=1)
        cprs_x_sd = np.std(total_cprs_x, axis=1) / np.sqrt(total_cprs_x.shape[1])

        d = {
            'rmse_x': rmse_x,
            'rmse_x_sd': rmse_x_sd,
            'cprs_x': cprs_x,
            'cprs_x_sd': cprs_x_sd
        }
        return d


def evaluate_flow(model, data_generator, batch_size, t0, mc_itr=50, real=False):
    with torch.no_grad():
        # sample-level rmse
        total_rmse_z0 = list()
        total_rmse_x = list()
        total_cprs_z0 = list()
        total_cprs_x = list()

        for chunk in range(data_generator.test_size // batch_size):
            data = data_generator.get_split('test', batch_size, chunk)

            x = data['measurements'][:t0]
            a = data['actions'][:t0]
            mask = data['masks'][:t0]
            z0 = data['latents'][0]
            if real:
                s = data['statics'][:t0]

            # Evaluate the goodness of point estimate
            if real:
                raise ValueError
            else:
                encoder_out = model.encoder(x, a, mask)
                mu, log_var, z0_hat, log_det_j, z0 = model.encoder.reparameterize(*encoder_out)
                x_hat, _ = model.decoder(z0_hat, data['actions'])

            x_hat = x_hat[t0:, ...]
            total_rmse_z0.append(torch.sum((z0[:, :data_generator.expert_dim] - z0_hat[:, :data_generator.expert_dim])**2, dim=1))

            # predicting future
            x_test = data['measurements'][t0:]
            mask_test = data['masks'][t0:]
            total_rmse_x.append(torch.sum((x_test - x_hat) ** 2 * mask_test, dim=(0, 2)) / torch.sum(mask_test, dim=(0, 2)))

            z_list = list()
            x_hat_list = list()

            for i in range(mc_itr):
                mu, log_var, z_, log_det_j, z0 = model.encoder.reparameterize(*encoder_out)
                if real:
                    x_hat, _ = model.decoder(z_, data['actions'], data['statics'])
                else:
                    x_hat, _ = model.decoder(z_, data['actions'])
                z_list.append(z_)
                x_hat_list.append(x_hat)

            # B, D, MC
            z_mat = torch.stack(z_list, dim=-1)

            # B, D
            z_cprs = np.zeros(z0[:, :data_generator.expert_dim].shape)
            for d1 in range(z0.shape[0]):
                for d2 in range(data_generator.expert_dim):
                    truth = z0[d1, d2].item()
                    pred = z_mat[d1, d2, :].cpu().numpy()
                    z_cprs[d1, d2] = ps.crps_ensemble(truth, pred)
            # B
            z_cprs = np.mean(z_cprs, axis=1)
            total_cprs_z0.append(z_cprs)

            # T0, B, D, MC
            x_hat_mat = torch.stack(x_hat_list, dim=-1)[t0:, ...]
            x_cprs = np.zeros(x_test.shape)
            for d1 in range(x_test.shape[0]):
                for d2 in range(x_test.shape[1]):
                    for d3 in range(x_test.shape[2]):
                        truth = x_test[d1, d2, d3].item()
                        pred = x_hat_mat[d1, d2, d3, :].cpu().numpy()
                        x_cprs[d1, d2, d3] = ps.crps_ensemble(truth, pred)
            x_cprs = np.mean(x_cprs, axis=(0, 2))
            total_cprs_x.append(x_cprs)

        total_rmse_z0 = torch.cat(total_rmse_z0)
        rmse_z0 = torch.sqrt(torch.mean(total_rmse_z0)).item()
        rmse_z0_sd = bootstrap_RMSE(total_rmse_z0)

        total_cprs_z0 = np.concatenate(total_cprs_z0)
        cprs_z0 = np.mean(total_cprs_z0)
        cprs_z0_sd = np.std(total_cprs_z0) / np.sqrt(len(total_cprs_z0))

        total_rmse_x = torch.cat(total_rmse_x)
        total_rmse_x = total_rmse_x[~torch.isnan(total_rmse_x)]
        rmse_x = torch.sqrt(torch.mean(total_rmse_x)).item()
        rmse_x_sd = bootstrap_RMSE(total_rmse_x)

        total_cprs_x = np.concatenate(total_cprs_x)
        cprs_x = np.mean(total_cprs_x)
        cprs_x_sd = np.std(total_cprs_x) / np.sqrt(len(total_cprs_x))


        print('rmse_z0,{:.4f},{:.4f}'.format(rmse_z0, rmse_z0_sd))
        print('rmse_x,{:.4f},{:.4f}'.format(rmse_x, rmse_x_sd))
        print('cprs_z0,{:.4f},{:.4f}'.format(cprs_z0, cprs_z0_sd))
        print('cprs_x,{:.4f},{:.4f}'.format(cprs_x, cprs_x_sd))

        return rmse_z0, rmse_z0_sd, cprs_z0, rmse_x, rmse_x_sd, cprs_x


def evaluate_ensemble(model_expert, model_ml, data_generator, batch_size, t0, mc_itr=50, weight_expert=1, weight_ml=1):
    with torch.no_grad():
        # sample-level rmse
        total_rmse_z0 = list()
        total_rmse_x = list()
        total_cprs_z0 = list()
        total_cprs_x = list()

        for chunk in range(data_generator.test_size // batch_size):
            data = data_generator.get_split('test', batch_size, chunk)

            x = data['measurements'][:t0]
            a = data['actions'][:t0]
            mask = data['masks'][:t0]
            z0 = data['latents'][0]

            # Evaluate the goodness of point estimate
            encoder_out = model_expert.encoder(x, a, mask)
            z0_hat = encoder_out[0]
            x_hat, _ = model_expert.decoder(z0_hat, data['actions'])

            encoder_out_ml = model_ml.encoder(x, a, mask)
            z0_hat_ml = encoder_out_ml[0]
            x_hat_ml, _ = model_ml.decoder(z0_hat_ml, data['actions'])

            x_hat = x_hat * weight_expert + x_hat_ml * weight_ml
            x_hat = x_hat[t0:, ...]

            total_rmse_z0.append(torch.sum((z0[:, :data_generator.expert_dim] - z0_hat[:, :data_generator.expert_dim])**2, dim=1))

            # predicting future
            x_test = data['measurements'][t0:]
            mask_test = data['masks'][t0:]
            total_rmse_x.append(torch.sum((x_test - x_hat) ** 2 * mask_test, dim=(0, 2)) / torch.sum(mask_test, dim=(0, 2)))

            z_list = list()
            x_hat_list = list()

            for i in range(mc_itr):
                z_ = model_expert.encoder.reparameterize(*encoder_out)
                x_hat, _ = model_expert.decoder(z_, data['actions'])

                z_ml_ = model_ml.encoder.reparameterize(*encoder_out_ml)
                x_hat_ml, _ = model_ml.decoder(z_ml_, data['actions'])

                x_hat = x_hat * weight_expert + x_hat_ml * weight_ml

                z_list.append(z_)
                x_hat_list.append(x_hat)

            # todo: this will break for flows

            # B, D, MC
            z_mat = torch.stack(z_list, dim=-1)

            # B, D
            z_cprs = np.zeros(z0[:, :data_generator.expert_dim].shape)
            for d1 in range(z0.shape[0]):
                for d2 in range(data_generator.expert_dim):
                    truth = z0[d1, d2].item()
                    pred = z_mat[d1, d2, :].cpu().numpy()
                    z_cprs[d1, d2] = ps.crps_ensemble(truth, pred)
            # B
            z_cprs = np.mean(z_cprs, axis=1)
            total_cprs_z0.append(z_cprs)

            # T0, B, D, MC
            x_hat_mat = torch.stack(x_hat_list, dim=-1)[t0:, ...]
            x_cprs = np.zeros(x_test.shape)
            for d1 in range(x_test.shape[0]):
                for d2 in range(x_test.shape[1]):
                    for d3 in range(x_test.shape[2]):
                        truth = x_test[d1, d2, d3].item()
                        pred = x_hat_mat[d1, d2, d3, :].cpu().numpy()
                        x_cprs[d1, d2, d3] = ps.crps_ensemble(truth, pred)
            x_cprs = np.mean(x_cprs, axis=(0, 2))
            total_cprs_x.append(x_cprs)

        total_rmse_z0 = torch.cat(total_rmse_z0)
        rmse_z0 = torch.sqrt(torch.mean(total_rmse_z0)).item()
        rmse_z0_sd = bootstrap_RMSE(total_rmse_z0)

        total_cprs_z0 = np.concatenate(total_cprs_z0)
        cprs_z0 = np.mean(total_cprs_z0)
        cprs_z0_sd = np.std(total_cprs_z0) / np.sqrt(len(total_cprs_z0))

        total_rmse_x = torch.cat(total_rmse_x)
        rmse_x = torch.sqrt(torch.mean(total_rmse_x)).item()
        rmse_x_sd = bootstrap_RMSE(total_rmse_x)

        total_cprs_x = np.concatenate(total_cprs_x)
        cprs_x = np.mean(total_cprs_x)
        cprs_x_sd = np.std(total_cprs_x) / np.sqrt(len(total_cprs_x))

        print('rmse_z0,{:.4f},{:.4f}'.format(rmse_z0, rmse_z0_sd))
        print('rmse_x,{:.4f},{:.4f}'.format(rmse_x, rmse_x_sd))
        print('cprs_z0,{:.4f},{:.4f}'.format(cprs_z0, cprs_z0_sd))
        print('cprs_x,{:.4f},{:.4f}'.format(cprs_x, cprs_x_sd))

        return rmse_z0, rmse_z0_sd, cprs_z0, rmse_x, rmse_x_sd, cprs_x


def evaluate_ensemble_horizon(model_expert, model_ml, data_generator, batch_size, t0, mc_itr=10, weight_expert=1, weight_ml=1):
    with torch.no_grad():
        # sample-level rmse
        total_rmse_x = list()
        total_cprs_x = list()

        for chunk in range(data_generator.test_size // batch_size):
            data = data_generator.get_split('test', batch_size, chunk)

            x = data['measurements'][:t0]
            a = data['actions'][:t0]
            mask = data['masks'][:t0]
            z0 = data['latents'][0]

            # Evaluate the goodness of point estimate
            encoder_out = model_expert.encoder(x, a, mask)
            z0_hat = encoder_out[0]
            x_hat, _ = model_expert.decoder(z0_hat, data['actions'])

            encoder_out_ml = model_ml.encoder(x, a, mask)
            z0_hat_ml = encoder_out_ml[0]
            x_hat_ml, _ = model_ml.decoder(z0_hat_ml, data['actions'])

            x_hat = x_hat * weight_expert + x_hat_ml * weight_ml
            x_hat = x_hat[t0:, ...]

            # predicting future
            x_test = data['measurements'][t0:]
            mask_test = data['masks'][t0:]
            total_rmse_x.append(torch.sum((x_test - x_hat) ** 2 * mask_test, dim=2) / torch.sum(mask_test, dim=2))

            z_list = list()
            x_hat_list = list()

            for i in range(mc_itr):
                z_ = model_expert.encoder.reparameterize(*encoder_out)
                x_hat, _ = model_expert.decoder(z_, data['actions'])

                z_ml_ = model_ml.encoder.reparameterize(*encoder_out_ml)
                x_hat_ml, _ = model_ml.decoder(z_ml_, data['actions'])

                x_hat = x_hat * weight_expert + x_hat_ml * weight_ml

                z_list.append(z_)
                x_hat_list.append(x_hat)

            # T0, B, D, MC
            x_hat_mat = torch.stack(x_hat_list, dim=-1)[t0:, ...]
            x_cprs = np.zeros(x_test.shape)
            for d1 in range(x_test.shape[0]):
                for d2 in range(x_test.shape[1]):
                    for d3 in range(x_test.shape[2]):
                        truth = x_test[d1, d2, d3].item()
                        pred = x_hat_mat[d1, d2, d3, :].cpu().numpy()
                        x_cprs[d1, d2, d3] = ps.crps_ensemble(truth, pred)
            x_cprs = np.mean(x_cprs, axis=2)
            total_cprs_x.append(x_cprs)

            # T, B
            total_rmse_x = torch.cat(total_rmse_x, dim=1)
            # T
            rmse_x = torch.sqrt(torch.nanmean(total_rmse_x, dim=1)).numpy()
            rmse_x_sd_list = []

            for i in range(rmse_x.shape[0]):
                rmse_x_sd_list.append(bootstrap_RMSE(total_rmse_x[i]))

            rmse_x_sd = np.array(rmse_x_sd_list)

            total_cprs_x = np.concatenate(total_cprs_x, axis=1)
            cprs_x = np.mean(total_cprs_x, axis=1)
            cprs_x_sd = np.std(total_cprs_x, axis=1) / np.sqrt(total_cprs_x.shape[1])

            d = {
                'rmse_x': rmse_x,
                'rmse_x_sd': rmse_x_sd,
                'cprs_x': cprs_x,
                'cprs_x_sd': cprs_x_sd
            }
            return d


def bootstrap_RMSE(err_sq):
    if type(err_sq) == np.ndarray:
        err_sq = torch.tensor(err_sq)
    rmse_list = []
    for i in range(500):
        new_err = err_sq[torch.randint(len(err_sq), err_sq.shape)]
        rmse_itr = torch.sqrt(torch.mean(new_err))
        rmse_list.append(rmse_itr.item())
    return np.std(np.array(rmse_list))
