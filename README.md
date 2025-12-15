# Robot-Trajectron

The official implementation of Robot Trajectron, which is accepted by ICRA2024: https://arxiv.org/pdf/2402.02499.pdf


### ✨ News
- 2025.10.25: We release our new work [Robot Trajectron V2](https://github.com/mousecpn/Robot-TrajectronV2)!
- 

![image](https://github.com/mousecpn/Robot-Trajectron/assets/46233799/c07bf977-ab2d-438f-8187-898d5efacd13)

![mnggiflab-compressed-demo_RT (1) (1)](https://github.com/mousecpn/Robot-Trajectron/assets/46233799/def7e71b-f0a4-45d4-b268-c9d327c78881)


### 🔧 Environment

 - numpy==1.24.4
 - tqdm==4.45.0
 - matplotlib==3.7.2
 - opencv-python==4.8.0.74
 - scikit-learn==0.22.1
 - scipy==1.10.1
 - seaborn==0.12.2
 - torch==2.0.0
 - CUDA==11.7
 - torchvision==0.15.1

### Dataset

Traj100k: https://github.com/mousecpn/Traj100k-Dataset.git

### 📈 Train

```
$ python main.py --eval_every 10 --vis_every 10 --preprocess_workers 0 --batch_size 256 --log_dir experiments/RobotTrajectron/models --train_epochs 100 --conf config/config.json --data_path /path/to/dataset
```

### 🎯 Test
```
$ python evalute.py --batch_size 256 --conf config/test_config.json --data_path /path/to/dataset --checkpoint /path/to/checkpoint
```

### Visualization
```
$ python visualization.py --conf config/test_config.json --data_path /path/to/dataset --checkpoint /path/to/checkpoint
```

### Shared Control

The code is in https://github.com/mousecpn/RT_ws.git



