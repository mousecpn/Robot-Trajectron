import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import pickle
import torch
from dataset.preprocessing import load_data_cartesian, TrajDataset, load_data_cartesian2
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from model.trajectron import Trajectron
from argument_parser import args
import json
from evaluate import visualize_distribution2d,visualize_distribution2d_running

def main():
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

    args.conf = 'config/test_config.json'
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval

    _, testData,  target_frequecy = load_data_cartesian('/home/pinhao/Desktop/Trajectron_for_robot/data.json', target_frequecy=10, min_length=20)

    hyperparams["frequency"] = target_frequecy

    trajectron = Trajectron(hyperparams,
                            None,
                            args.device)
    model = torch.load(args.checkpoint)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()
    count = 1
    for data in testData:
        if count != 13:
            count+=1
            continue
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # plt.xlabel('x label') 
        # plt.ylabel('y label')
        # ax.set_axis_off()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])
        # ax.axes.get_zaxis().set_visible(False)
        plt.ion()
        # data = data[::2,:]
        steps = data.shape[0]-10
        x_range = (0.0, 15)
        y_range = (-7.5, 7.5)
        z = data[:,2].min()
        last_prod=None
        curve = None

        for j in range(steps):
            first_history_index = torch.LongTensor(np.array([0])).cuda()
            x = data[j:j+8,:9]
            y = data[j+8:j+12,:9]
            # ph = data.shape[0]-(j+8)
            ph = 12
            std = np.array([3,3,3,2,2,2,1,1,1])
            # std = np.array([1,1,1,1,1,1,1,1,1])

            rel_state = np.zeros_like(x[0])
            rel_state[0:3] = np.array(x)[-1, 0:3]

            x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
            y_st = np.where(np.isnan(y), np.array(np.nan), y / std)
            x_t = torch.tensor(x, dtype=torch.float).unsqueeze(0).cuda()
            y_t = torch.tensor(y, dtype=torch.float).unsqueeze(0).cuda()
            x_st_t = torch.tensor(x_st, dtype=torch.float).unsqueeze(0).cuda()
            y_st_t = torch.tensor(y_st, dtype=torch.float).unsqueeze(0).cuda()
            batch = (first_history_index, x_t, y_t[...,3:6], x_st_t, y_st_t[...,3:6])
            try:
                with torch.no_grad():
                    ################# most likely ##############################
                    y_dist, predictions = trajectron.predict2(batch,
                                            z_T=0.05*15,
                                            num_samples=1,
                                            z_mode=True,
                                            gmm_mode=True,
                                            all_z_sep=False,
                                            full_dist=False,
                                            dist=True)
            except:
                pass


            vis_data = data[:j+8,:9]
            ax.plot3D(vis_data[:,0], vis_data[ :,1], vis_data[:,2], 'green')
            ax.scatter3D(vis_data[::2,0], vis_data[::2,1], vis_data[::2,2], s=5, c='green')
            vis_pred = predictions[0]#.detach().cpu().numpy()
            vis_pred = np.concatenate((data[j+7:j+8,:3].reshape(1,1,3), vis_pred),axis=1)
            if curve is not None:
                curve.remove()
                dist_print.remove()
            last_prod, dist_print = visualize_distribution2d_running(ax, y_dist, x_range, y_range, z, None, print=True)
            curve, = ax.plot3D(vis_pred[0, :,0], vis_pred[0, :,1], vis_pred[0, :,2], 'red')
            # plt.savefig('imgs/traj_step{}_grid{}.png'.format(j,2))
            plt.pause(0.01)
            plt.ioff()
            # plt.show()

            # break
            # for s in range(predictions.shape[0]):
            #     vis_pred = predictions[s]#.detach().cpu().numpy()
            #     vis_pred = np.concatenate((data[j+7:j+8,:3].reshape(1,1,3), vis_pred),axis=1)
            #     ax.plot3D(vis_pred[0, :,0], vis_pred[0, :,1], vis_pred[0, :,2], 'red')
            # plt.show()

            # break
        # visualize_distribution2d_running(ax,y_dist,x_range, y_range, z, last_prod)
        data = data
        ax.plot3D(data[:,0], data[ :,1], data[:,2], 'green')
        ax.scatter3D(data[::2,0], data[::2,1], data[::2,2], s=5, c='green')
        
        ax.set_title('3D line plot')
        plt.show()
    return


if __name__=="__main__":
    main()