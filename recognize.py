import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from collections import deque
import winsound

# =========================
# CONFIG
# =========================
EMBEDDINGS_PATH = "embeddings.pkl"

DISTANCE_THRESHOLD = 1.05
CONFIDENCE_GAP = 0.015

SMOOTHING_WINDOW = 7
UNKNOWN_TOLERANCE = 8

FRAME_SKIP = 2
RESIZE_SCALE = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# ALERT SOUND
# =========================
def play_alert():
    winsound.Beep(1000, 500)

# =========================
# LOAD MODELS
# =========================
mtcnn = MTCNN(
    keep_all=True,
    image_size=160,
    margin=40,
    min_face_size=40,
    device=device
)

resnet = InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# =========================
# LOAD EMBEDDINGS
# =========================
with open(EMBEDDINGS_PATH, "rb") as f:
    raw_db = pickle.load(f)

all_embeddings = []
labels = []

for person, emb_list in raw_db.items():
    for emb in emb_list:
        all_embeddings.append(emb)
        labels.append(person)

all_embeddings = np.array(all_embeddings)

# =========================
# ALERT CONTROL
# =========================
detected_people = set()

# =========================
# EMBEDDING
# =========================
def get_embedding(face):

    if face.shape[0] < 20 or face.shape[1] < 20:
        return None

    face = cv2.resize(
        face,
        (160,160),
        interpolation=cv2.INTER_CUBIC
    )

    # CLAHE enhancement
    lab = cv2.cvtColor(
        face,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    face = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    face = face.astype('float32') / 255.0

    face = np.transpose(face, (2,0,1))

    tensor = torch.tensor(face).unsqueeze(0).to(device)

    with torch.no_grad():

        emb = resnet(tensor)

        emb = emb / torch.norm(emb)

    return emb.cpu().numpy()[0]

# =========================
# MATCHING
# =========================
def match_face(embedding):

    distances = np.linalg.norm(
        all_embeddings - embedding,
        axis=1
    )

    sorted_idx = np.argsort(distances)

    best_idx = sorted_idx[0]
    second_idx = sorted_idx[1]

    best_dist = distances[best_idx]
    second_dist = distances[second_idx]

    best_name = labels[best_idx]

    if (
        best_dist < DISTANCE_THRESHOLD and
        (second_dist - best_dist) > CONFIDENCE_GAP
    ):

        return best_name, best_dist

    else:

        return "Unknown", best_dist

# =========================
# RECOGNITION
# =========================
def recognize_frame(frame):

    global last_valid_name
    global unknown_counter

    small = cv2.resize(
        frame,
        (0,0),
        fx=RESIZE_SCALE,
        fy=RESIZE_SCALE
    )

    rgb = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2RGB
    )

    # brightness correction
    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.4,
        beta=35
    )

    boxes, probs = mtcnn.detect(rgb)

    if boxes is None:
        return frame

    for box, prob in zip(boxes, probs):

        if prob < 0.90:
            continue

        x1, y1, x2, y2 = map(int, box)

        # Scale back
        x1 = int(x1 / RESIZE_SCALE)
        y1 = int(y1 / RESIZE_SCALE)
        x2 = int(x2 / RESIZE_SCALE)
        y2 = int(y2 / RESIZE_SCALE)

        face = frame[y1:y2, x1:x2]

        try:

            emb = get_embedding(face)

            if emb is None:
                continue

            raw_name, distance = match_face(emb)

            name = raw_name

        except:
            continue

        # =========================
        # ALERT
        # =========================
        if name != "Unknown":

            # Beep only once per appearance
            if name not in detected_people:

                play_alert()

                print(f"[ALERT] MISSING PERSON DETECTED: {name}")

                detected_people.add(name)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,0,255),
                3
            )

            cv2.putText(
                frame,
                "MISSING PERSON DETECTED",
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

            cv2.putText(
                frame,
                f"{name} ({distance:.2f})",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )
    return frame

# =========================
# START WEBCAM
# =========================
cap = cv2.VideoCapture(0)

# Higher webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

last_frame = None

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # PROCESS ONLY EVERY NTH FRAME
    if frame_count % FRAME_SKIP == 0:

        frame = recognize_frame(frame)

        last_frame = frame.copy()

    else:

        # Reuse previous processed frame
        if last_frame is not None:
            frame = last_frame.copy()

    cv2.imshow(
        "Missing Person Detection System",
        frame
    )

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()