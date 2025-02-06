
import torch
import matplotlib.pyplot as plt

handle_trajectories = torch.load('dataset/action/handle_trajectory_0_eval.pth')
handle_trajectories = handle_trajectories[:, :, :-1, :]
print(f"handle_trajectories = {handle_trajectories.shape}")

handle_trajectory = handle_trajectories[0, 0, :, :]
print(f"handle_trajectory = {handle_trajectory}")

handle_trajectory_0th_elements = [handle_trajectory[0].item() for handle_trajectory in handle_trajectory if handle_trajectory is not None]
handle_trajectory_1th_elements = [handle_trajectory[1].item() for handle_trajectory in handle_trajectory if handle_trajectory is not None]

indexs = list(range(len(handle_trajectory_0th_elements)))

# 軌道（正規化）描画用
x_sim = [0.0 for _ in range(len(indexs))]
y_sim = [0.0 for _ in range(len(indexs))]
x_sim_value = 0.0
y_sim_value = 0.0
for i in range(1, len(indexs)):
    x_sim_value += handle_trajectory_0th_elements[i]
    y_sim_value += handle_trajectory_1th_elements[i]
    x_sim[i] = x_sim_value
    y_sim[i] = y_sim_value

# 理論値の軌道（正規化後）
plt.figure(figsize=(8, 6))
# plt.plot(y_theory, x_theory, marker='o', color='blue', label='theoretical value')
plt.plot(handle_trajectory_1th_elements, handle_trajectory_0th_elements, marker='o', color='red', label='corrected label')
plt.axis("equal")
plt.gca().invert_yaxis()
plt.title("trajectory", fontsize=14)
plt.xlabel("y [-]", fontsize=12)
plt.ylabel("x [-]", fontsize=12)
# plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
# plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig("軌跡.png")
plt.close()