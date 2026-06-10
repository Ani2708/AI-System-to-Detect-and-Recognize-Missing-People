import os
import cv2
from facenet_pytorch import MTCNN
import torch

input_dir = "faces_clustered/Anirudh_raw"
output_dir = "faces_clustered/Anirudh"

os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=False,
    device=device
)

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    save_path = os.path.join(output_dir, img_name)

    # IMPORTANT: let MTCNN handle everything
    mtcnn(rgb, save_path=save_path)

print("Faces extracted correctly!")