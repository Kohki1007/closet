import torch
import numpy as np
import random
from torchvision import transforms

transform = transforms.Resize((224, 224))

ob_current = torch.zeros([100, 5, 40, 224, 224])
for i in range(20):
    ob_current[i * 5 : (i+1)*5] = torch.load(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_ob_{i * 5}.pt").clone()

# torch.save(ob_current, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_ob.pt")
st_current = torch.load(f"/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_st.pt")

current_image = torch.cat([ob_current, st_current], dim = 0)
print(current_image.shape)
# torch.save(current_image, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_rem_1204.pt")

target_image = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/target.pt")

def rand_ints_nodup(a, b, k):
    ns = []
    while len(ns) < k:
        n = np.random.randint(a, b)
        if not n in ns:
            ns.append(n)
    nss = np.sort(np.array(ns))
    return nss

# num = np.array(range(24000))
nums = np.array(range(24000))

val_idx = rand_ints_nodup(0, 24000, 2000)
train_idx = np.delete(nums, val_idx)

val_idx = val_idx.tolist()
train_idx = train_idx.tolist()
# print(current.shape)
current = current_image.view(-1, 1, 224, 224)
current_train = current[train_idx]
current_val = current[val_idx]
# torch.save(current_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_train.pt")
# torch.save(current_val, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/current_val.pt")

current_train = []
current_val = []
current = []

target = target_image.view(-1, 1, 256, 256)
# print(target.shape)
target_train = target[train_idx]
target_val = target[val_idx]

# torch.save(target_train, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/target_train.pt")
# torch.save(target_val, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset_remover/target_val.pt")