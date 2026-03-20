import cv2
import face_recognition
from pathlib import Path

VIDEO_PATH = "Video_1/Class_8_cctv_video_1.mov"
OUT_DIR = Path("raw_faces")
OUT_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

known_encodings = []
saved_count = 0
frame_idx = 0

FRAME_SKIP = 60   # detect every 2 sec
TOLERANCE = 0.5

print("Extracting unique identities...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP == 0:

        # Resize (balanced for CCTV)
        small = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locs = face_recognition.face_locations(rgb_small, model="hog")

        if face_locs:
            encodings = face_recognition.face_encodings(rgb_small, face_locs)

            for (loc, enc) in zip(face_locs, encodings):
                top, right, bottom, left = loc

                # Ignore tiny faces
                if (bottom - top) < 40:
                    continue

                # Duplicate check (FAST)
                if known_encodings:
                    matches = face_recognition.compare_faces(
                        known_encodings, enc, tolerance=TOLERANCE
                    )
                    if True in matches:
                        continue

                # Scale back
                scale = 1 / 0.75
                top, right, bottom, left = map(int, [
                    top * scale, right * scale, bottom * scale, left * scale
                ])

                face_img = frame[
                    max(0, top-20):bottom+20,
                    max(0, left-20):right+20
                ]

                # Save
                cv2.imwrite(str(OUT_DIR / f"person_{saved_count}.jpg"), face_img)
                known_encodings.append(enc)
                saved_count += 1

                print(f"New person found → {saved_count}")

    frame_idx += 1

cap.release()
print(f"Done. Total unique people: {saved_count}")