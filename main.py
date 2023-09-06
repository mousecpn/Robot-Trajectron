import torch
import numpy as np
from torch.utils.data import DataLoader
from model.trajectron import Trajectron
from tqdm import tqdm
from model.mgcvae import MultimodalGenerativeCVAE
import json
from argument_parser import args
import os
from tensorboardX import SummaryWriter
import pathlib
import time
from model.model_registrar import ModelRegistrar
from dataset.preprocessing import load_data_cartesian, TrajDataset, load_data_cartesian2
from torch.utils.data._utils.collate import default_collate
import torch.nn as nn
import torch.optim as optim
import evaluation
from evaluation import compute_ade
import evaluate
import matplotlib.pyplot as plt

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


    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval

    best_ade = 1000
    
    trainData, testData, target_frequecy = load_data_cartesian2(args.data_path, 10, 20, test_size=0.1, aug=True)
    traindataset = TrajDataset(trainData,max_history_length=8, min_future_timesteps=12, eval=False)
    testdataset = TrajDataset(testData, max_history_length=8, min_future_timesteps=12, eval=True)
    train_dataloader = DataLoader(traindataset,
                                    collate_fn=default_collate,
                                    pin_memory=True,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=16)
    
    eval_dataloader = DataLoader(testdataset,
                                    collate_fn=default_collate,
                                    pin_memory=True,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0)
    
    hyperparams["frequency"] = target_frequecy


    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        log_writer = SummaryWriter(log_dir=model_dir)

    trajectron = Trajectron(hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_annealing_params()
    trajectron.model.train()



    optimizer = optim.Adam(trajectron.model.parameters(), lr=hyperparams['learning_rate'])
    # Set Learning Rate
    if hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=hyperparams['learning_decay_rate'])


    curr_iter = 0
    for epoch in range(1, args.train_epochs + 1):
        trajectron.model.to(args.device)
        pbar = tqdm(train_dataloader, ncols=80)
        for batch in pbar:
            (first_history_index, x_t, y_t, x_st_t, y_st_t) = batch
            batch = (first_history_index, x_t, y_t[...,3:6], x_st_t, y_st_t[...,3:6])
            trajectron.set_curr_iter(curr_iter)
            trajectron.step_annealers()
            optimizer.zero_grad()
            # try:
            train_loss = trajectron.train_loss(batch)
            # except:
            #     continue
            pbar.set_description(f"Epoch {epoch},  L: {train_loss.item():.2f}")
            train_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model.parameters(), hyperparams['grad_clip'])
            optimizer.step()

            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler.step()

            if not args.debug:
                log_writer.add_scalar(f"train/learning_rate",
                                        lr_scheduler.get_last_lr()[0],
                                        curr_iter)
                log_writer.add_scalar(f"train/loss", train_loss, curr_iter)
            curr_iter += 1
            
        print("learning_rate:",lr_scheduler.get_last_lr()[0])

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            trajectron.model.to(args.device)
            trajectron.model.eval()
            with torch.no_grad():
                # Calculate evaluation loss
                eval_loss_list = []
                print(f"Starting Evaluation @ epoch {epoch}")
                pbar = tqdm(eval_dataloader, ncols=80)
                ade = []
                fde = []
                for batch in pbar:
                    (first_history_index, x_t, y_t, x_st_t, y_st_t) = batch
                    batch = (first_history_index, x_t, y_t[...,3:6], x_st_t, y_st_t[...,3:6])
                    eval_loss = trajectron.eval_loss(batch)
                    
                    pbar.set_description(f"Epoch {epoch}, L: {eval_loss.item():.2f}")
                    eval_loss_list.append({'nll': [eval_loss]})
                    # predictions = trajectron.predict(batch,
                    #                         ph,
                    #                         num_samples=20,
                    #                         z_mode=False,
                    #                         gmm_mode=False,
                    #                         full_dist=False)
                    predictions = trajectron.predict(batch,
                                        ph,
                                        num_samples=20,
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=True,
                                        full_dist=False)
                    
                    batch_ade = np.min(evaluation.compute_ade(predictions, y_t[...,0:3].detach().cpu().numpy()),axis=0)
                    batch_fde = np.min(evaluation.compute_fde(predictions, y_t[...,0:3].detach().cpu().numpy()),axis=0)
                    # ax = plt.axes(projection='3d')
                    # visualization.plot_trajectories(ax, predictions, x_t[0,:,0:3].detach().cpu().numpy() ,y_t[0,:,0:3].detach().cpu().numpy())


                    ade.append(batch_ade)
                    fde.append(batch_fde)
                ade = np.mean(np.concatenate(ade,axis=0))*1000/15
                fde = np.mean(np.concatenate(fde,axis=0))*1000/15
            if ade < best_ade:
                best_ade = ade
                # model_registrar.save_models(epoch)
                model_save = trajectron.model
                torch.save(model_save.node_modules, "checkpoints/epoch{}|100k|aug|ade{:.2f}.pth".format(epoch,ade))
                # torch.save(model_save.node_modules, "checkpoints/epoch{}|aug|ade{:.2f}.pth".format(epoch,ade))
            print("ade:", ade)
            print("fde:", fde)
                    
            trajectron.model.train()


    return


if __name__=="__main__":
    main()

# python main.py --eval_every 10 --vis_every 10 --preprocess_workers 0 --batch_size 256 --log_dir experiments/RobotTrajectron/models --train_epochs 100 --conf config/config.json