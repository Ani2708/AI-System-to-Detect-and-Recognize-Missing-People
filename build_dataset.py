import os
import cv2
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch

# CONFIG
CHOKEPOINT_DIR = "chokepoint"
OUTPUT_DIR = "faces"
FRAME_SKIP = 5
MIN_CONFIDENCE = 0.95

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(keep_all=True, device=device)

def get_sequence_name(folder):
    parts = folder.split("_")
    return "_".join(parts[:2])  # P2E_S1_C3.1 → P2E_S1

def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unknown_dir = os.path.join(OUTPUT_DIR, "unknown")
    os.makedirs(unknown_dir, exist_ok=True)

    for folder in os.listdir(CHOKEPOINT_DIR):
        if "groundtruth" in folder.lower():
            continue

        folder_path = os.path.join(CHOKEPOINT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing {folder}")
        sequence_name = get_sequence_name(folder)

        images = sorted(os.listdir(folder_path))

        for i, img_name in enumerate(tqdm(images)):
            if not img_name.endswith(".jpg"):
                continue
            if i % FRAME_SKIP != 0:
                continue

            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            boxes, probs = detector.detect(rgb)

            if boxes is None:
                continue

            for j, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < MIN_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(int, box)

                face = rgb[y1:y2, x1:x2]

                try:
                    face = cv2.resize(face, (160, 160))
                except:
                    continue

                filename = f"{sequence_name}_{img_name[:-4]}_{j}.jpg"
                save_path = os.path.join(unknown_dir, filename)

                cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

    print("Dataset created!")

if __name__ == "__main__":
    process()