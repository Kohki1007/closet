import torch

closet_rot = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/closet_props/real/closet_rot.pt")
handle_trajectory = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__/trajectory.pth")
handle_trajectory_1 = torch.load("/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__/trajectory_1.pth")
handle_trajectory = torch.cat([handle_trajectory, handle_trajectory_1], dim = 0)
action = torch.zeros(160, 5, 40, 6)

print(closet_rot.shape)

for env in range(closet_rot.shape[0]):
    for position in range(closet_rot.shape[1]):
        for step in range(closet_rot.shape[2]-1):
            start_point = closet_rot[env, position, step]

            if int(start_point) + 4 < 181:
                action[env, position, step, :6] = handle_trajectory[env, position, int(start_point) + 4] - handle_trajectory[env, position, int(start_point)]
            elif int(start_point) + 4 > 180:
                print(step)
                action[env, position, step, :6] = handle_trajectory[env, position, 180] - handle_trajectory[env, position, int(start_point)]
            # output[step, 6:] = handle_trajectory[int(start_point)]
            action[env, position, step, 2] = 0

            action[env, position, step, :3] /= torch.sqrt(torch.sum(action[env, position, step, :3]**2))
            action[env, position, step, 3:] /= torch.sqrt(torch.sum(action[env, position, step, 3:]**2))

torch.save(action, "/home/engawa/py_ws/visual_servo/src/network/sim2real/dataset/real__/action.pth")