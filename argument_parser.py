import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    type=str,
                    default='../config/config1.json')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)




# Data Parameters
parser.add_argument("--data_dir",
                    help="what dir to look in for data",
                    type=str,
                    default='../experiments/processed')

parser.add_argument("--data_path",
                    help="json",
                    type=str,
                    default="/home/pinhao/Desktop/franka_sim/traj_fre20_noisy_100k.json")

parser.add_argument("--checkpoint",
                    help="the checkpoint file",
                    type=str,
                    default="/home/pinhao/Desktop/Trajectron_for_robot/checkpoints/epoch29|finetune100k|ade15.04.pth")

parser.add_argument("--log_dir",
                    help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str,
                    default='../experiments/logs')

parser.add_argument("--log_tag",
                    help="tag for the log folder",
                    type=str,
                    default='Traj100k')

parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')

parser.add_argument("--eval_device",
                    help="what device to use during evaluation",
                    type=str,
                    default=None)

# Training Parameters
parser.add_argument("--train_epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=1)

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)

parser.add_argument('--eval_batch_size',
                    help='evaluation batch size',
                    type=int,
                    default=256)

parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--vis_every',
                    help='how often to visualize during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--save_every',
                    help='how often to save during training, never if None',
                    type=int,
                    default=1)
args = parser.parse_args()
