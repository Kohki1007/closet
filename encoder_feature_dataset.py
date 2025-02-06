import torch
import torch.nn as nn
import numpy as np
# from torchviz import make_do
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from livelossplot import PlotLosses
import torch.nn.init as init
from predict_targetimage_action import Autoencoder, set_requires_grad
from loss import Loss
from torchvision import transforms
import torch.nn.functional as F
import random


import sys 

def r_ints_nodup(a, b, k):
            ns = []
            while len(ns) < k:
                n = np.random.randint(a, b)
                if not n in ns:
                    ns.append(n)
            nss = np.sort(np.array(ns))
            return nss


nums = np.array(range(2000))
test_idx = r_ints_nodup(0, 2000, 200)
train_idx = np.delete(nums, test_idx, 0)
feature_list = torch.zeros(2000, 512, 8, 8)


base = 'src/network/log/all_0823'
args = sys.argv

# args = [0, "init", "rgbd_attension"]

BATCH_SIZE = 16 #2の11乗
LEARNING_RATE = 0.0001  # 学習率： 0.03
LEARNING_RATE_D = 0.0002
LEARNING_RATE_G = 0.0002
REGULARIZATION = 0.03  # 正則化率： 0.03


best_accuracy = 0
device2 = torch.device("cuda")
device2 = torch.device("cpu")
rot = torch.load("src/network/predict_action/dataset/predict_action/closet_props/closet_rot.pth")
rot = rot[:40, :50].reshape(2000)
torch.save(rot, "/home/engawa/py_ws/visual_servo/src/network/dataset/dataset_real/dataset_estimate_rotation/output.pt")

path_input_init = "src/network/predict_action/dataset/predict_action/dataset/middle_depth_rgb_attension.pth"
path_input_current = "src/network/deleteUR/rgbe_rgbd/dataset/image_t.pth"
path_output_target = "src/network/predict_action/dataset/predict_action/dataset/target_depth_rgb_attension.pth"
path_output_delete = "src/network/deleteUR/rgbe_rgbd/dataset/image_t-.pth"
path_action = "src/network/predict_action/dataset/predict_action/dataset/output_with_vec_dev.pth"


x_init = torch.load(path_input_init).to(device2)[:, 0].unsqueeze(1)
x_init = x_init.expand(-1, 50, -1, -1, -1)
x_init = x_init.reshape(2000, 327680)
# x_current = torch.load(path_input_current).to(device2).view(2000, -1)
x_current = torch.load(path_input_current).to(device2).permute(1, 2, 0, 3, 4, 5)
x_current = x_current.reshape(2000, 3276800)

model = Autoencoder()
model = model.to("cuda")


def step(train_X_init, train_X_current, i):
    train_X_init = train_X_init.view(-1, 5, 256, 256)
    train_X_current = train_X_current.view(-1, 10, 5, 256, 256)
    train_X_current_all = train_X_current[:, random.randint(0, 9)]
    train_X_current = train_X_current_all[:, :4]
    train_X_current_attension = train_X_current_all[:, 4].to('cuda')

    
    # 訓練モードに設定
    model.eval()
    with torch.no_grad():
    # pred_y = model(train_X_init, train_X_target) # 出力結果
        train_X_init = train_X_init.to('cuda')
        train_X_current = train_X_current.to('cuda')


        feature = model.get_feature(train_X_init, train_X_current, train_X_current_attension.unsqueeze(1))

    feature_list[i*100:(i+1)*100] = feature



targetG_weight = torch.load("/home/engawa/py_ws/visual_servo/src/network/pytorch-CycleGAN-and-pix2pix/checkpoints/rgbd_d_0807/latest_net_G.pth", map_location='cuda')
deleteG_weight = torch.load("/home/engawa/py_ws/visual_servo/src/network/pytorch-CycleGAN-and-pix2pix/checkpoints/delete_ur_many/latest_net_G.pth", map_location='cuda')
# deleteG_weight = torch.load("src/network/pytorch-CycleGAN-and-pix2pix/checkpoints/delete_ur_many/latest_net_G.pth", map_location='cuda')
action_weight = torch.load("src/network/log/0825/weight/weights_last.pth", map_location='cuda')

model.generate_target.module.load_state_dict(targetG_weight)
model.delete_ur.module.load_state_dict(deleteG_weight)
model.predict_action.load_state_dict(action_weight)

for i in range(20):
    init = x_init[i*100:(i+1)*100]
    current = x_current[i*100:(i+1)*100]
    step(init, current, i)

torch.save(feature_list, "/home/engawa/py_ws/visual_servo/src/network/dataset/dataset_real/dataset_estimate_rotation/input.pt")
print('Finished Training')
# print(model.state_dict())
# print(model.state_dict())  # 学習後のパラメーターの情報を表示