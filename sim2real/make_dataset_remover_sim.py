import torch
import numpy as np

target1 = torch.load("dataset/remover_many/output_0_1214.pt")
target2 = torch.load("dataset/remover_many/output_0_1214.pt")

target = torch.cat([target1.clone(), target2.clone()], dim = 0)

target1 = []
target2 = []

current_handle = torch.load("dataset/remover_sim/current_handle.pt")
current_st = torch.load("dataset/remover_sim/current_st.pt")

for i in range(2):
    currant_ob_element = torch.load(f"dataset/remover_sim/current_ob_{i}.pt")
    if i == 0:
        current_ob = currant_ob_element.clone()
    else:
        current_ob = torch.cat([current_ob, currant_ob_element], dim = 0)

curent = torch.cat([current_handle.clone(), current_st.clone(), current_ob.clone()], dim = 0)

torch.save(curent, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_sim/current.pt")

current_handle = []
current_st = []
current_ob = []

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

val_idx = rand_ints_nodup(0, 32000, 4000)
train_idx = np.delete(nums, val_idx)
curent = curent.view(32000, 1, 224, 224)
target = target.view(32000, 1, 224, 224)

torch.save(target[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_sim/target_train.pt")
torch.save(target[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_sim/target_val.pt")

torch.save(curent[train_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_sim/current_train.pt")
torch.save(curent[val_idx], "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/remover_sim/current_val.pt")
