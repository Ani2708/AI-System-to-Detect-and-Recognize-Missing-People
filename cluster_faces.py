import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from facenet_pytorch import InceptionResnetV1
import torch
import shutil
from tqdm import tqdm

FACES_DIR = "faces/unknown"
OUTPUT_DIR = "faces_clustered"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face):
    face = cv2.resize(face, (160,160))
    face = face.astype('float32') / 255.0
    face = np.transpose(face, (2,0,1))

    tensor = torch.tensor(face).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(tensor)

    return emb.cpu().numpy()[0]

images = []
embeddings = []

print("Generating embeddings...")

for img_name in tqdm(os.listdir(FACES_DIR)):
    img_path = os.path.join(FACES_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    emb = get_embedding(img)

    images.append(img_name)
    embeddings.append(emb)

embeddings = np.array(embeddings)

print("Clustering...")

clustering = DBSCAN(metric='euclidean', eps=0.6, min_samples=3)
labels = clustering.fit_predict(embeddings)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name, label in zip(images, labels):
    if label == -1:
        folder = "unknown"
    else:
        folder = f"person_{label}"

    folder_path = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(folder_path, exist_ok=True)

    src = os.path.join(FACES_DIR, img_name)
    dst = os.path.join(folder_path, img_name)

    shutil.copy(src, dst)

print("Clustering complete!")