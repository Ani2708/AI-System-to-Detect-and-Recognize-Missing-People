import cv2
import numpy as np
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
from collections import deque
import winsound

EMBEDDINGS_PATH = "embeddings.pkl"

VIDEO_1 = "chokepoint_trim1.mp4"
VIDEO_2 = "chokepoint_trim2.mp4"


DISTANCE_THRESHOLD = 1.12
CONFIDENCE_GAP = 0.015

SMOOTHING_WINDOW = 7
UNKNOWN_TOLERANCE = 8

FRAME_SKIP = 2
RESIZE_SCALE = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
# TRACKING
# =========================
last_seen = {}

camera_presence = {}

MOVE_CONFIRM_FRAMES = 2

camera_alerts = {
    "CAM 1": set(),
    "CAM 2": set(),
}

# =========================
# SMOOTHING
# =========================
history = deque(maxlen=SMOOTHING_WINDOW)

def smooth_prediction(pred):

    history.append(pred)

    counts = {
        p: history.count(p)
        for p in history
    }

    return max(counts, key=counts.get)

# =========================
# UNKNOWN SUPPRESSION
# =========================
last_valid_name = None
unknown_counter = 0

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
def recognize_frame(frame, cam_name):

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
            name = smooth_prediction(raw_name)

            if name != "Unknown":
                last_valid_name = name
                unknown_counter = 0
            else:
                unknown_counter += 1
                if (
                    last_valid_name is not None and
                    unknown_counter < UNKNOWN_TOLERANCE
                ):
                    name = last_valid_name
        except:
            continue

        # =========================
        # ALERT + TRACKING
        # =========================
        if name != "Unknown":

            now = datetime.now().strftime("%H:%M:%S")

            # Beep only once
            if name not in camera_alerts[cam_name]:

                play_alert()

                camera_alerts[cam_name].add(name)

            # Presence tracking
            if name not in camera_presence:

                camera_presence[name] = {
                    "camera": cam_name,
                    "count": 1
                }

            else:

                if camera_presence[name]["camera"] == cam_name:

                    camera_presence[name]["count"] += 1

                else:

                    camera_presence[name]["camera"] = cam_name
                    camera_presence[name]["count"] = 1

            # Confirm movement
            if camera_presence[name]["count"] >= MOVE_CONFIRM_FRAMES:

                if name not in last_seen:

                    print(
                        f"[ALERT] MISSING PERSON DETECTED: "
                        f"{name} at {cam_name} at {now}"
                    )

                else:

                    last_cam, _ = last_seen[name]

                    if last_cam != cam_name:

                        print(
                            f"[MOVE] {name} moved "
                            f"from {last_cam} → "
                            f"{cam_name} at {now}"
                        )

                last_seen[name] = (cam_name, now)

            # DRAW
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
# VIDEO STREAMS
# =========================
cap1 = cv2.VideoCapture(VIDEO_1)
cap2 = cv2.VideoCapture(VIDEO_2)

cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

frame_count = 0

# Store processed frames
last_frame1 = None
last_frame2 = None

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break

    # PROCESS ONLY EVERY NTH FRAME
    if frame_count % FRAME_SKIP == 0:

        if ret1:
            frame1 = recognize_frame(frame1, "CAM 1")
            last_frame1 = frame1.copy()

        if ret2:
            frame2 = recognize_frame(frame2, "CAM 2")
            last_frame2 = frame2.copy()

    else:

        if last_frame1 is not None and ret1:
            frame1 = last_frame1.copy()

        if last_frame2 is not None and ret2:
            frame2 = last_frame2.copy()

    if ret1:

        cv2.putText(
            frame1,
            "Camera 1",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

    if ret2:

        cv2.putText(
            frame2,
            "Camera 2",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )

    frames = []

    for f in [frame1 if ret1 else None,
              frame2 if ret2 else None]:

        if f is not None:
            frames.append(f)

    if len(frames) > 0:

        h = min(frame.shape[0] for frame in frames)

        resized_frames = []

        for frame in frames:

            resized = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * h / frame.shape[0]),
                    h
                )
            )

            resized_frames.append(resized)

        combined = np.hstack(resized_frames)

        cv2.imshow(
            "Multi-Camera View",
            combined
        )

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()

cv2.destroyAllWindows()

# =========================
# FINAL TRACKING DATA
# =========================
print("\n=== FINAL TRACKING DATA ===")

for person, (cam, time) in last_seen.items():

    print(
        f"{person} last seen "
        f"at {cam} at {time}"
    )