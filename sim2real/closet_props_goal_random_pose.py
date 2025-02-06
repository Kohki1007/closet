import torch
import numpy as np
import random

# rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_rot.pt")

# print(rot[0])

closet_position = torch.zeros(160, 5, 2)
closet_rot = torch.zeros(160, 5, 41)
closet_size = torch.zeros(160, 2)
wall_position = torch.zeros(160)
camera_pose = torch.zeros(160, 3)
y_max = [0.28, 0.2, 0.1, 0.02, -0.07]
# y_min = [-0.45, -0.4, -0.3, -0.25, -0.18]
y_min = [-0.48, -0.4, -0.3, -0.22, -0.13]
y_base = -0.1
x_base = -0.95

for i in range(closet_position.shape[0]):
    id = i%4
    wall_position[i] = random.uniform(-1.6, -1.3) 
    if id == 0:
        closet_size[i, 1] = random.uniform(0.8, 1.0)
    elif id == 1:
        closet_size[i, 1] = random.uniform(1.0, 1.2)
    elif id == 2:
        closet_size[i, 1] = random.uniform(1.2, 1.4)
    elif id == 3:
        closet_size[i, 1] = random.uniform(1.4, 1.6)
    
    closet_size[i, 0] = random.uniform(0.75, 1.15)

    camera_pose[i, 0] = random.uniform(-15, 15) 
    # print(y_max[id+1])
    # print(y_min_old)

    # y_max_old = y_max[id+1]
    # y_min_old = y_min[id+1]

    sita_max = torch.arctan(torch.abs((torch.tensor(y_max[id+1]+ 0.1)/-0.95)))
    sita_max += camera_pose[i]*3.14/180
    sita_min = torch.arctan(torch.abs((torch.tensor(y_min[id+1]+ 0.1)/-0.95)))
    sita_min -= camera_pose[i]*3.14/180

    new_max = []
    # print(y_max[id+1])




    for j in range(closet_position.shape[1]):
        closet_position[i, j, 0] = random.uniform(wall_position[i] + 0.9, wall_position[i] + 1.15)

        dy_max = torch.abs(closet_position[i, j, 0].clone()) *  torch.tan(sita_max)
        dy_min = torch.abs(closet_position[i, j, 0].clone())*  torch.tan(sita_min)

        # print(-dy_min -0.1, dy_max - 0.1)
        # print(dy_min, dy_max)

        closet_position[i, j, 1] = random.uniform(-dy_min -0.1, dy_max - 0.1)

        # if id == 0:
        # #     closet_position[i, j, 1] = random.uniform(-0.55, 0.21)
        # # elif id == 1:
        #     closet_position[i, j, 1] = random.uniform(-0.55, 0.126)
        # elif id == 1:
        #     closet_position[i, j, 1] = random.uniform(-0.55, 0.03)
        # elif id == 2:
        #     closet_position[i, j, 1] = random.uniform(-0.55, -0.03)
        # elif id == 3:
        #     closet_position[i, j, 1] = random.uniform(-0.55, -0.09)

        closet_rot[i, j, :-1] = torch.tensor(sorted(random.sample(range(170), k=160)))
    closet_rot[i, :, -1] = 180
    

    # wall_position = random.uniform(-2.0, -1.5) 





# torch.save(closet_position, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_posistion.pt")
# torch.save(closet_rot, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_rot.pt")
# torch.save(closet_size, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/closet_size.pt")
# torch.save(wall_position, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/generator_1214/wall_position.pt")

torch.save(closet_position, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/camera_pose_random/closet_posistion.pt")
torch.save(closet_rot, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/camera_pose_random/closet_rot.pt")
torch.save(closet_size, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/camera_pose_random/closet_size.pt")
torch.save(wall_position, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/camera_pose_random/wall_position.pt")
torch.save(camera_pose, "/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/camera_pose_random/camera_pose.pt")
