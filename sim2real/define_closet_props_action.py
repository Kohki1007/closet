import torch
import random

closet_position = torch.zeros(100, 10, 2)
closet_rot = torch.zeros(100, 10, 41)
closet_size = torch.zeros(100)

# [:50, :5, :40]

for i in range(closet_position.shape[0]):
    for j in range(closet_position.shape[1]):
        closet_position[i, j, 0] = random.uniform(-0.95, -0.65)
        
        id = i%5
        if id == 0:
            closet_position[i, j, 1] = random.uniform(-0.55, 0.15)
        elif id == 1:
            closet_position[i, j, 1] = random.uniform(-0.55, 0.08)
        elif id == 2:
            closet_position[i, j, 1] = random.uniform(-0.55, 0)
        elif id == 3:
            closet_position[i, j, 1] = random.uniform(-0.55, -0.05)
        elif id == 4:
            closet_position[i, j, 1] = random.uniform(-0.55, -0.1)
        closet_rot[i, j, :-1] = torch.tensor(sorted(random.sample(range(180), k=40)))
        # closet_size[i] = random.uniform(0.75, 1.15)

closet_rot[:, :, -1] = 180

torch.save(closet_position, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/remover_sim/closet_posistion.pt")
torch.save(closet_rot, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/remover_sim/closet_rot.pt")
torch.save(closet_size, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/remover_sim/closet_size.pt")