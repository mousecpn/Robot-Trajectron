from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
from tqdm import tqdm
import evaluate
import evaluation
import matplotlib.pyplot as plt
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from tensorboardX import SummaryWriter
from dataset.preprocessing import load_data_cartesian,TrajDataset,load_data_cartesian2
from torch.utils.data import DataLoader

def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=1,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    # for node in histories_dict:
    history = histories_dict
    future = futures_dict
    predictions = prediction_dict
    future = np.concatenate((history[-1:,:],future),axis=0)
    node_circle_size = np.sqrt(np.mean((history[1:] - history[:-1,:])**2)) * node_circle_size

    ax.plot3D(history[:, 0], history[:, 1], history[:, 2], 'k--')

    for sample_num in range(prediction_dict.shape[0]):

        if kde and predictions.shape[0] >= 50:
            line_alpha = 0.2
            for t in range(predictions.shape[2]):
                sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                            ax=ax, shade=True, shade_lowest=False,
                            color=np.random.choice(cmap), alpha=0.8)

        ax.plot3D(predictions[sample_num, batch_num, :, 0], predictions[sample_num, batch_num, :, 1],predictions[sample_num, batch_num, :, 2],
                color=cmap[1],
                linewidth=line_width, alpha=line_alpha)

        ax.plot3D(future[:, 0],
                future[:, 1],
                future[:, 2],
                'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        # Current Node Position
        # circle = plt.Circle((history[-1, 0],
        #                      history[-1, 1]),
        #                     node_circle_size,
        #                     facecolor='g',
        #                     edgecolor='k',
        #                     lw=circle_edge_width,
        #                     zorder=3)
        # ax.add_artist(circle)

    # ax.axis('equal')
    

def visualize_distribution(ax,
                           prediction_distribution_dict,
                           map=None,
                           pi_threshold=0.05,
                           **kwargs):
    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    for node, pred_dist in prediction_distribution_dict.items():
        if pred_dist.mus.shape[:2] != (1, 1):
            return

        means = pred_dist.mus.squeeze().cpu().numpy()
        covs = pred_dist.get_covariance_matrix().squeeze().cpu().numpy()
        pis = pred_dist.pis_cat_dist.probs.squeeze().cpu().numpy()

        for timestep in range(means.shape[0]):
            for z_val in range(means.shape[1]):
                mean = means[timestep, z_val]
                covar = covs[timestep, z_val]
                pi = pis[timestep, z_val]

                if pi < pi_threshold:
                    continue

                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = patches.Ellipse(mean, v[0], v[1], 180. + angle, color='blue' if node.type.name == 'VEHICLE' else 'orange')
                ell.set_edgecolor(None)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(pi/10)
                ax.add_artist(ell)
        plt.show()

def visualize_distribution2d(ax,
                           pred_dist,
                           pi_threshold=0.05,
                           **kwargs):

    means = pred_dist.mus
    pis = pred_dist.pis_cat_dist.probs

    n_component, _, timesteps,_, dim = means.shape
    n = 0
    mode_t_list = []
    t = timesteps - 1
    # for t in range(timesteps):
    nt_gmm = pred_dist.get_for_node_at_time(n, t)
    x_min = means[:, n, t, :, 0].min() - 1
    x_max = means[:, n, t, :, 0].max() + 1
    y_min = means[:, n, t, :, 1].min() - 1
    y_max = means[:, n, t, :, 1].max() + 1
    z = means[:, n, t, :, 2].min()
    XYZ = torch.meshgrid([torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01),torch.FloatTensor([z])])
    search_grid = torch.stack(XYZ, dim=2
                                ).view(-1, 3).float().to(means.device)

    score = torch.exp(nt_gmm.log_prob(search_grid))
    prob = torch.sum(score.squeeze()*pis[:,n,t,:].reshape(-1,1),dim=0).reshape(XYZ[0].shape)
    X = XYZ[0][...,0].cpu().numpy()
    Y = XYZ[1][...,0].cpu().numpy()
    prob = prob[...,0].cpu().numpy()
    ax.plot_surface(X, Y, prob, cmap=plt.get_cmap('rainbow'))

def visualize_distribution2d_running(ax,
                                pred_dist,
                                x_range,
                                y_range,
                                z,
                                last_prod=None,
                                print_=True,
                                **kwargs):

    means = pred_dist.mus
    pis = pred_dist.pis_cat_dist.probs

    n_component, _, timesteps,_, dim = means.shape
    n = 0
    mode_t_list = []
    t = timesteps - 1
    # for t in range(timesteps):
    nt_gmm = pred_dist.get_for_node_at_time(n, t)
    x_min = x_range[0]
    x_max = x_range[1]
    y_min = y_range[0]
    y_max = y_range[1]
    for k in range(1):
        XYZ = torch.meshgrid([torch.arange(x_min+k*0.02*15, x_max+k*0.02*15, 0.05*15), torch.arange(y_min+k*0.02*15, y_max+k*0.02*15, 0.05*15),torch.FloatTensor([z])])
        search_grid = torch.stack(XYZ, dim=2).view(-1, 3).float().to(means.device)

        score = torch.exp(nt_gmm.log_prob(search_grid))
        prob = torch.sum(score,dim=0).reshape(XYZ[0].shape)
        # print("latent funtion:", prob.sum().cpu().numpy())
        prob = prob/prob.sum()
        X = XYZ[0][...,0].cpu().numpy()
        Y = XYZ[1][...,0].cpu().numpy()
        prob = prob[...,0].cpu().numpy()
    if last_prod is not None:
        prob = np.sqrt(prob*last_prod)
    if print_:
        dist_print = ax.plot_surface(X, Y, prob, cmap=plt.get_cmap('rainbow'))
        return prob, dist_print
    return prob

if __name__=='__main__':
    from argument_parser import args

    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    if args.eval_device is None:
        args.eval_device = torch.device('cpu')

    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)


    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    best_ade = 1000

    # trainData, testData, target_frequecy = load_data_cartesian('./traj_fre20_noisy_20000.json', 10, 20)
    trainData, testData, target_frequecy = load_data_cartesian2(args.data_path, 10, 20, test_size=0.1, aug=False)
    testdataset = TrajDataset(testData, max_history_length=8, min_future_timesteps=12, eval=False)
    
    eval_dataloader = DataLoader(testdataset,
                                    pin_memory=True,
                                    batch_size=256,
                                    shuffle=True,
                                    num_workers=0)
    
    hyperparams["frequency"] = target_frequecy


    log_writer = None
    model_dir = None

    trajectron = Trajectron(hyperparams,
                            log_writer,
                            args.device)
    # model = torch.load("/home/pinhao/Desktop/Trajectron_for_robot/checkpoints/epoch80|ade17.62.pth")
    # model = torch.load("/home/pinhao/Desktop/Trajectron_for_robot/checkpoints/epoch40|aug|ade20.59.pth")
    model = torch.load(args.checkpoint)
    # model = torch.load("/home/pinhao/Desktop/Trajectron_for_robot/checkpoints/epoch10|aug|ade10.95.pth")
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()

    #################################
    #           EVALUATION          #
    #################################
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    with torch.no_grad():
        # Calculate evaluation loss
        eval_loss_list = []
        pbar = tqdm(eval_dataloader, ncols=80)
        ade = []
        fde = []
        for batch in pbar:
            # fig = plt.figure()
            (first_history_index, x_t, y_t, x_st_t, y_st_t) = batch
            batch = (first_history_index, x_t, y_t[...,3:6], x_st_t, y_st_t[...,3:6])
            eval_loss = trajectron.eval_loss(batch)
            
            pbar.set_description(f"L: {eval_loss.item():.2f}")
            eval_loss_list.append({'nll': [eval_loss]})
            # _, predictions = trajectron.predict(batch,
            #                         ph,
            #                         num_samples=20,
            #                         z_mode=False,
            #                         gmm_mode=True,
            #                         all_z_sep=True,
            #                         full_dist=False,
            #                         dist=True)
            predictions = trajectron.predict(batch,
                                        ph,
                                        num_samples=1,
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=False,
                                        full_dist=False)
            
            batch_ade = np.min(evaluation.compute_ade(predictions, y_t[...,0:3].detach().cpu().numpy()),axis=0)
            batch_fde = np.min(evaluation.compute_fde(predictions, y_t[...,0:3].detach().cpu().numpy()),axis=0)
            # ax = plt.axes(projection='3d')
            # visualize_distribution2d(ax,y_dist)
            # plot_trajectories(ax, predictions, x_t[0,:,0:3].detach().cpu().numpy() ,y_t[0,:,0:3].detach().cpu().numpy())
            # plt.show()

            ade.append(batch_ade)
            fde.append(batch_fde)
        ade = np.mean(np.concatenate(ade,axis=0))*1000/15
        fde = np.mean(np.concatenate(fde,axis=0))*1000/15
        print("ade:", ade)
        print("fde:", fde)

