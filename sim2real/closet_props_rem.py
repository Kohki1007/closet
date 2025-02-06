import torch
import numpy as np
import random

# rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_rot_object.pt")

# print(rot[0])

closet_position = torch.zeros(120, 5, 2)
closet_rot = torch.zeros(120, 5, 40)
closet_size = torch.zeros(120)

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

        closet_rot[i, j] = torch.tensor(sorted(random.sample(range(170), k=40)))
    closet_size[i] = random.uniform(0.75, 1.15)




torch.save(closet_position[:100], "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_rem/closet_posistion_ob.pt")
torch.save(closet_rot[:100], "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_rem/closet_rot_ob.pt")
torch.save(closet_size[:100], "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_rem/closet_size_ob.pt")

torch.save(closet_position[100:], "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_rem/closet_posistion_st.pt")
torch.save(closet_rot[100:], "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_rem/closet_rot_st.pt")
torch.save(closet_size[100:], "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_rem/closet_size_st.pt")
