import torch
import numpy as np
import random
from torchvision import transforms

transform = transforms.Resize((224, 224))

# ob_current = torch.zeros([100, 5, 40, 224, 224])
# for i in range(20):
#     ob_current[i * 5 : (i+1)*5] = torch.load(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/current_ob_{i * 5}.pt").clone()

# # torch.save(ob_current, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/current_ob.pt")
# st_current = torch.load(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/current_st.pt")

# action = torch.load("dataset/dataset_action/action_1208_2.pt")
# print(action.shape)

current_image = torch.load("dataset/dataset_action/current_action.pt")
size_id = [random.randint(0, 99) for _ in range(80)]
position_id = [random.randint(0, 14) for _ in range(10)]
step_id = [random.randint(0, 59) for _ in range(40)]
current_image = current_image[size_id]
current_image = current_image[:, position_id]
current_image = current_image[:, :, step_id]
# print(current_image.shape)
current_image = transform(current_image.view(-1, 1, 256, 256))
# print(current_image.shape)
# torch.save(current_image, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/current_rem_1204.pt")

target_image = torch.load("dataset/dataset_action/current_action.pt")
target_image = target_image[size_id]
target_image = target_image[:, position_id]
target_image = target_image[:, :, step_id]
print(target_image.shape)
target_image = transform(target_image.view(-1, 1, 256, 256))

action = torch.load("dataset/dataset_action/action_1208_2.pt")
action = action[size_id]
action = action[:, position_id]
action = action[:, :, step_id]
print(action.shape)


def rand_ints_nodup(a, b, k):
    ns = []
    while len(ns) < k:
        n = np.random.randint(a, b)
        if not n in ns:
            ns.append(n)
    nss = np.sort(np.array(ns))
    return nss

# num = np.array(range(24000))
nums = np.array(range(32000))

val_idx = rand_ints_nodup(0, 32000, 3000)
train_idx = np.delete(nums, val_idx)

val_idx = val_idx.tolist()
train_idx = train_idx.tolist()
# print(current.shape)
current = current_image.view(-1, 1, 224, 224)
current_train = current[train_idx]
current_val = current[val_idx]
torch.save(current_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/current_train.pt")
torch.save(current_val, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/current_val.pt")

current_train = []
current_val = []
current = []

target = target_image.view(-1, 1, 224, 224)
# print(target.shape)
target_train = target[train_idx]
target_val = target[val_idx]

action = action.view(-1, 6)
# print(target.shape)
action_train = action[train_idx]
action_test = action[val_idx]

torch.save(target_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/target_train.pt")
torch.save(target_val, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/target_val.pt")

torch.save(action_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/action_train.pt")
torch.save(action_test, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/action_val.pt")
