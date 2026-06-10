import os
import cv2
import numpy as np
import pickle
from facenet_pytorch import InceptionResnetV1
import torch

DATASET_DIR = "faces_clustered"
OUTPUT_FILE = "embeddings.pkl"

TRAIN_SPLIT = 0.8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face):
    if face.shape[0] < 20 or face.shape[1] < 20:
        return None

    # Resize (important for consistency)
    face = cv2.resize(face, (160,160), interpolation=cv2.INTER_CUBIC)

    # Convert to LAB for lighting correction
    lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (better than normal histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
    face = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Normalize
    face = face.astype('float32') / 255.0
    face = np.transpose(face, (2,0,1))

    tensor = torch.tensor(face).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(tensor)
        emb = emb / torch.norm(emb)

    return emb.cpu().numpy()[0]

database = {}

np.random.seed(42)

print("Building embeddings (TRAIN SPLIT ONLY)...")

for person in os.listdir(DATASET_DIR):

    if person == "unknown":
        continue

    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)
    np.random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_idx]

    embeddings = []

    for img_name in train_images:
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            emb = get_embedding(rgb)
            embeddings.append(emb)
        except:
            continue

    if len(embeddings) > 0:
        database[person] = embeddings

# Save embeddings
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(database, f)

print("Embeddings saved (train set only).")