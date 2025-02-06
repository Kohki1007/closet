import torch
import numpy as np

target = torch.load("dataset/trick/target.pt")
current = torch.load("dataset/trick/current.pt")
rot = torch.load("closet_props/trick/closet_rot.pt")
rot = rot[:, :, 1:]


def rand_ints_nodup(a, b, k):
    ns = []
    while len(ns) < k:
        n = np.random.randint(a, b)
        if not n in ns:
            ns.append(n)
    nss = np.sort(np.array(ns))
    return nss

# num = np.array(range(24000))
nums = np.array(range(30000))

val_idx = rand_ints_nodup(0, 30000, 3000)
train_idx = np.delete(nums, val_idx)
curent = current.view(30000, 1, 224, 224)
target = target.view(30000, 1, 224, 224)
rot = rot.reshape(30000, 1)

torch.save(target[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/trick/target_train.pt")
torch.save(target[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/trick/target_test.pt")

torch.save(curent[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/trick/current_train.pt")
torch.save(curent[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/trick/current_test.pt")

torch.save(rot[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/trick/rot_train.pt")
torch.save(rot[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/trick/rot_test.pt")
