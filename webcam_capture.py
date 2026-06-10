import cv2
import os

save_dir = "faces_clustered/Anirudh_raw"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
print("Press SPACE to capture, Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  # SPACE to capture
        filename = os.path.join(save_dir, f"img_{count}.jpg")
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()