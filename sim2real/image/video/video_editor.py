import cv2
import os

image_path_list = [1, 2, 3, 4, 5]
# 画像が保存されているフォルダ
for id in image_path_list:
    image_folder = f'triming_{id}'
    output_video = f'{id}.mp4'

    # 画像ファイルの取得（ソートして順番に並べる）
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # または 'XVID'（AVIの場合）
    fps = 10  # フレームレート（1秒あたりのフレーム数）
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        frame = cv2.resize(frame, (width, height))
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

print(f"動画 {output_video} が作成されました。")