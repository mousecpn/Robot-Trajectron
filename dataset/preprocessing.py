import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from torch.utils.data._utils.collate import default_collate

def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x

def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt
    # dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=0.0) / dt

    return dx

def derivatives_of(x, dt=1, radian=False):
    timestep, dim = x.shape
    dxs = []
    for d in range(dim):
        dxs.append(derivative_of(x[:,d],dt))
    dxs = np.stack(dxs,axis=-1)
    return dxs


def load_data_cartesian2(path, target_frequecy, min_length, test_size=0.3, aug=True):
    # trainData = {}
    trainData = []
    testData = []
    with open(path,'rb') as f:
        data = pickle.load(f)
    ee_log = data["ee_log"]
    frequency = data["frequency"]
    dt = 1./target_frequecy
    scale = 15
    # dt = 1./frequency

    stride = int(frequency//target_frequecy)
    train_set, test_set = train_test_split(ee_log, test_size=test_size, random_state=42)
    if aug == False:
        tr_num = 1
    else:
        tr_num = 7
    # training dataset
    for l in range(len(train_set)):
        cur_sequence = train_set[l]*scale
        for s in range(stride):
            idx_list = np.array(range(s, cur_sequence.shape[0], stride))
            # term = cur_sequence[:,:3]
            term_ori = cur_sequence[idx_list,:3]
            if term_ori.shape[0] < min_length:
                term_ori = cur_sequence

            for tr in range(0,tr_num):
                term = transform(term_ori, tr)
                vel_term = derivatives_of(term, dt=dt)
                acc_term = derivatives_of(vel_term, dt=dt)
                term = np.concatenate((term,vel_term,acc_term),axis=-1)
                # term = term[idx_list,:]

                if term.shape[0] < min_length:
                    continue
                # else:
                #     term = term[:length,:]
                trainData.append(term)
    # trainData = np.stack(trainData)

    for l in range(len(test_set)):
        cur_sequence = test_set[l]*scale
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        # term = cur_sequence[:,:3]
        term = cur_sequence[idx_list,:3]
        vel_term = derivatives_of(term, dt=dt)
        acc_term = derivatives_of(vel_term, dt=dt)
        term = np.concatenate((term,vel_term,acc_term),axis=-1)
        for j in range(term.shape[0]//min_length):
            term_j = term[j*min_length:(j+1)*min_length]
            if term_j.shape[0] < min_length:
                continue

            testData.append(term_j)
        # if term.shape[0] < min_length:
        #     continue

        # testData.append(term)
    return trainData, testData, target_frequecy

def load_data_cartesian(path, target_frequecy, min_length, test_size=0.3):
    # trainData = {}
    trainData = []
    testData = []
    with open(path,'rb') as f:
        data = pickle.load(f)
    ee_log = data["ee_log"]
    frequency = data["frequency"]
    dt = 1./target_frequecy
    scale = 15
    # dt = 1./frequency

    stride = int(frequency//target_frequecy)
    train_set, test_set = train_test_split(ee_log, test_size=test_size, random_state=42)

    # training dataset
    for l in range(len(train_set)):
        cur_sequence = train_set[l]*scale
        # for s in range(stride):
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        # term = cur_sequence[:,:3]
        term = cur_sequence[idx_list,:3]
        vel_term = derivatives_of(term, dt=dt)
        acc_term = derivatives_of(vel_term, dt=dt)
        term = np.concatenate((term,vel_term,acc_term),axis=-1)
        # term = term[idx_list,:]

        if term.shape[0] < min_length:
            continue
        # else:
        #     term = term[:length,:]
        trainData.append(term)
    # trainData = np.stack(trainData)

    for l in range(len(test_set)):
        cur_sequence = test_set[l]*scale
        cur_sequence += (np.random.rand(cur_sequence.shape[0],cur_sequence.shape[1])-0.5)*0.2
        # cur_sequence = transform(cur_sequence, 6)
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        # term = cur_sequence[:,:3]
        term = cur_sequence[idx_list,:3]
        # term = term[::2,:]
        vel_term = derivatives_of(term, dt=dt)
        acc_term = derivatives_of(vel_term, dt=dt)
        term = np.concatenate((term,vel_term,acc_term),axis=-1)
        if term.shape[0] < min_length:
            continue
        testData.append(term)
        # for j in range(term.shape[0]//min_length):
        #     term_j = term[j*min_length:(j+1)*min_length]
        #     # term = term[idx_list,:]
        #     if term_j.shape[0] < min_length:
        #         continue
        #     # else:
        #     #     term = term[:length,:]
        #     testData.append(term_j)
    # testData = np.stack(testData)
    return trainData, testData, target_frequecy

def transform(cur_sequence, trans_idx, scale=15):
    # speed up 2x
    # if np.random.rand() > 0.8:
    #     cur_sequence = cur_sequence[::2,:]
    # noise
    if trans_idx == 1:
        idx = np.random.randint(0,3)
        cur_sequence[:,idx] *= 0.1
    # reverse x
    # elif trans_idx == 2:
    #     x_seq = cur_sequence[:,0]
    #     x_seq = x_seq.tolist()
    #     x_seq.reverse()
    #     cur_sequence[:,0] = np.array(x_seq)
    #     if np.random.rand() > 0.5:
    #         cur_sequence = transform(cur_sequence, 6, scale=15)
    # reverse y
    if trans_idx == 2:
        y_seq = cur_sequence[:,1]
        y_seq = y_seq.tolist()
        y_seq.reverse()
        cur_sequence[:,1] = np.array(y_seq)
        if np.random.rand() > 0.5:
            trans_idx = 4
    # reverse z
    # elif trans_idx == 5:
    #     z_seq = cur_sequence[:,2]
    #     z_seq = z_seq.tolist()
    #     z_seq.reverse()
    #     cur_sequence[:,2] = np.array(z_seq)
    # mirror y
    if trans_idx == 3:
        cur_sequence[:,1] *= -1
        if np.random.rand() > 0.5:
            trans_idx = 4
    # strech x
    if trans_idx == 4:
        x_seq = cur_sequence[:,0]
        x_seq = (((x_seq-x_seq.min())/(x_seq.max()-x_seq.min()))*np.random.uniform(0.1,0.7)+np.random.uniform(0.1,0.3))*scale
        cur_sequence[:,0] = np.array(x_seq)
        if np.random.rand() > 0.5:
            trans_idx = 6
    # strech y
    if trans_idx == 5:
        y_seq = cur_sequence[:,1]
        y_seq = (y_seq/np.abs(y_seq).max())*np.random.uniform(0.0,0.4)*scale
        cur_sequence[:,1] = np.array(y_seq)
        if np.random.rand() > 0.5:
            trans_idx = 6
    # strech z
    if trans_idx == 6:
        z_seq = cur_sequence[:,2]
        z_seq = (((z_seq-z_seq.min())/(z_seq.max()-z_seq.min()))*np.random.uniform(0.3,0.7)+np.random.uniform(-0.2, 0.04))*scale
        # z_seq = (z_seq)*np.random.uniform(0.8,1.1)
        cur_sequence[:,2] = np.array(z_seq)
    if np.random.rand() > 0.5:
        cur_sequence += (np.random.rand(cur_sequence.shape[0],cur_sequence.shape[1])-0.5)*0.2
    return cur_sequence

class TrajDataset(Dataset):
    def __init__(self, Data, max_history_length, min_future_timesteps, eval=False):
        self.data = Data
        # self.data = torch.FloatTensor(Data)
        self.max_ht = max_history_length
        self.min_ft = min_future_timesteps
        # self.inseq = self.data[:, :in_length, :]
        # self.outseq = self.data[:, in_length:, :]
        self.eval = eval

        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        seq = self.data[index]
        if self.eval:
            t = np.array(8)
        else:
            t = np.random.choice(np.arange(3, len(seq)-self.min_ft), replace=False)
            # t = np.array(8)
        timestep_range_x = np.array([t - self.max_ht, t])
        timestep_range_y = np.array([t, t + self.min_ft])
        first_history_index = (self.max_ht - t).clip(0)

        length = timestep_range_x[1] - timestep_range_x[0]

        data_array = seq[max(timestep_range_x[0],0):timestep_range_x[1]]
        x = np.full((length, data_array.shape[1]), fill_value=np.nan)
        x[first_history_index:length] = data_array.copy()

        y = seq[max(timestep_range_y[0],0):timestep_range_y[1]] # velocity

        # x = seq[max(timestep_range_x[0],0):timestep_range_x[1]+1]
        # y = seq[timestep_range_y[0]:timestep_range_y[1],3:6] # velocity

        std = np.array([3,3,3,2,2,2,1,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])
        # std = np.array([2,2,2,2,2,2,1,1,1])


        rel_state = np.zeros_like(x[0])
        rel_state[0:3] = np.array(x)[-1, 0:3]

        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), y / std)
        x_t = torch.tensor(x, dtype=torch.float)
        y_t = torch.tensor(y, dtype=torch.float)
        x_st_t = torch.tensor(x_st, dtype=torch.float)
        y_st_t = torch.tensor(y_st, dtype=torch.float)


        return first_history_index, x_t, y_t, x_st_t, y_st_t


if __name__=="__main__":
    trainData, testData, target_frequecy = load_data_cartesian2('traj_fre20_noisy_20000.json', 10, 20)
    traindataset = TrajDataset(trainData, max_history_length=8, min_future_timesteps=12, eval=False)
    train_dataloader = DataLoader(traindataset,
                                    collate_fn=default_collate,
                                    pin_memory=True,
                                    batch_size=12,
                                    shuffle=True,
                                    num_workers=0)
    for data in train_dataloader:
        print(data)