import os
from PIL import Image
import torch
from torchvision import transforms

# 画像を読み込む関数
def load_image_as_tensor(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    return image_tensor

def load_images_with_prefix(directory, prefix):
    tensors = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(directory, filename)
            try:
                tensor = load_image_as_tensor(image_path)
                tensor = torch.where(tensor == 0, 1, tensor)
                tensors.append(tensor)
                print(f"Loaded: {filename} -> Tensor Shape: {tensor.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    tensors = torch.stack(tensors)
    return tensors

# 実行例
directory = "dataset/dataset_real/goal_image/input"  # 画像ファイルがあるディレクトリ
prefix = "depth_image_resize"  # 検索する接頭辞

# 実行
tensors = load_images_with_prefix(directory, prefix)

torch.save(tensors, "dataset/dataset_real/goal_image/dataset/real_input.pt")