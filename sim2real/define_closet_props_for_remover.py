import torch
import random

closet_position_ob = torch.zeros([50, 5, 2])
closet_position_st = torch.zeros([10, 5, 2])

closet_rot_ob = torch.zeros([50, 5, 40])
closet_rot_st = torch.zeros([10, 5, 40])

closet_size_ob = torch.zeros([50])
closet_size_st = torch.zeros([50])

for i in range(closet_position_ob.shape[0]):
    for j in range(closet_position_ob.shape[1]):
        closet_position_ob[i, j, 0] = random.uniform(-0.95, -0.65)
        
        id = i%5
        if id == 0:
            closet_position_ob[i, j, 1] = random.uniform(-0.55, 0.15)
        elif id == 1:
            closet_position_ob[i, j, 1] = random.uniform(-0.55, 0.08)
        elif id == 2:
            closet_position_ob[i, j, 1] = random.uniform(-0.55, 0)
        elif id == 3:
            closet_position_ob[i, j, 1] = random.uniform(-0.55, -0.05)
        elif id == 4:
            closet_position_ob[i, j, 1] = random.uniform(-0.55, -0.1)
        closet_rot_ob[i, j] = torch.tensor(sorted(random.sample(range(170), k=40)))
        closet_size_ob[i] = random.uniform(0.75, 1.15)

        if i < 10:
            closet_position_st[i, j, 0] = random.uniform(-0.95, -0.65)
            if id == 0:
                closet_position_st[i, j, 1] = random.uniform(-0.55, 0.15)
            elif id == 1:
                closet_position_st[i, j, 1] = random.uniform(-0.55, 0.08)
            elif id == 2:
                closet_position_st[i, j, 1] = random.uniform(-0.55, 0)
            elif id == 3:
                closet_position_st[i, j, 1] = random.uniform(-0.55, -0.05)
            elif id == 4:
                closet_position_st[i, j, 1] = random.uniform(-0.55, -0.1)
            closet_rot_st[i, j] = torch.tensor(sorted(random.sample(range(170), k=40)))
            closet_size_st[i] = random.uniform(0.75, 1.15)

torch.save(closet_position_ob, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_posistion_rem_ob.pt")
torch.save(closet_position_st, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_posistion_rem_st.pt")
torch.save(closet_rot_ob, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_rot_rem_ob.pt")
torch.save(closet_rot_st, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_rot_rem_st.pt")
torch.save(closet_size_ob, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_size_rem_ob.pt")
torch.save(closet_size_st, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/closet_size_rem_st.pt")