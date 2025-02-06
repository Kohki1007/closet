import torch

handle_trajectory = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_action/handle_trajectory.pth")
closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props_action/closet_rot.pt")

print(closet_rot.shape)

action = torch.zeros([100, 15, 60, 6])

# for size in range(closet_rot.shape[0]):
#     for position in range(closet_rot.shape[1]):
#         for step in range(closet_rot.shape[2]-1):

for size in range(100):
    for position in range(closet_rot.shape[1]):
        for step in range(closet_rot.shape[2]-1):
            rot_number = closet_rot[size, position, step]

            # print(rot_number)

            action[size, position, step, :6] = handle_trajectory[size, position, int(rot_number) + 6] - handle_trajectory[size, position, int(rot_number)]
            # action[size, step, 6:] = handle_trajectory[size, int(rot_number)]
            action[size, position, step, 2] = 0
            # print(f"before ===== {action[size, position, step, :3]}")

            action[size, position, step, :3] /= torch.sqrt(torch.sum(action[size, position, step, :3]**2))
            action[size, position, step, 3:] /= torch.sqrt(torch.sum(action[size, position, step, 3:]**2))
            # print(f"after ===== {action[size, position, step, :3]}")

            if torch.any(torch.isnan(action[size, position, step, 3:])):
                print(f"size ===== {size}")
                print(f"position ===== {position}")
                print(f"step ===== {step}")
            # print(torch.any(torch.isnan(action[size, position, step, 3:])))

            if torch.any(torch.isnan(action)):
                print(f"size ===== {size}")
                print(f"position ===== {position}")
                print(f"step ===== {step}")

torch.save(action, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/dataset_action/action_1208_2.pt")
print("complete")