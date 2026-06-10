import cv2
import numpy as np
import pickle
import os
from facenet_pytorch import InceptionResnetV1
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

TEST_DIR = "faces_clustered"
EMBEDDINGS_PATH = "embeddings.pkl"

TRAIN_SPLIT = 0.8
THRESHOLD = 0.8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings
with open(EMBEDDINGS_PATH, "rb") as f:
    database = pickle.load(f)

def get_embedding(face):
    face = cv2.resize(face, (160,160))
    face = face.astype('float32') / 255.0
    face = np.transpose(face, (2,0,1))

    tensor = torch.tensor(face).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(tensor)
        emb = emb / torch.norm(emb)

    return emb.cpu().numpy()[0]

# Matching
def match_face(embedding):
    best_match = "Unknown"
    min_dist = float("inf")

    for person, emb_list in database.items():
        for db_emb in emb_list:
            dist = np.linalg.norm(embedding - db_emb)

            if dist < min_dist:
                min_dist = dist
                best_match = person

    if min_dist < THRESHOLD:
        return best_match
    else:
        return "Unknown"

# =========================
# TEST LOOP
# =========================
y_true = []
y_pred = []

np.random.seed(42)

print("Testing model...\n")

for person in os.listdir(TEST_DIR):
    person_path = os.path.join(TEST_DIR, person)

    if not os.path.isdir(person_path):
        continue

    if person == "unknown":
        continue

    images = os.listdir(person_path)
    np.random.shuffle(images)

    split_idx = int(len(images) * TRAIN_SPLIT)
    test_images = images[split_idx:]

    for img_name in test_images:
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            emb = get_embedding(rgb)
            pred = match_face(emb)

            y_true.append(person)
            y_pred.append(pred)

            print(f"Actual: {person}, Predicted: {pred}")

        except:
            continue

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_true, y_pred)

print("\n=== PERFORMANCE METRICS ===")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
labels = list(set(y_true + y_pred))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved as 'confusion_matrix.png'")

from sklearn.metrics import precision_score, recall_score, f1_score

# Use macro average (recommended)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8,6))
plt.bar(metrics, values)

# Add values on top
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)

plt.ylim(0, 1.1)
plt.title("Model Performance Metrics")
plt.ylabel("Score")

plt.savefig("performance_metrics.png")
plt.show()

print("\nPerformance graph saved as 'performance_metrics.png'")