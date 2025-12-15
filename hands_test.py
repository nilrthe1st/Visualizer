import cv2
import mediapipe as mp
import math

FINGER_GAPS = [
    ("T-I", 4, 8),    # thumb tip to index tip
    ("I-M", 8, 12),   # index tip to middle tip
    ("M-R", 12, 16),  # middle tip to ring tip
    ("R-P", 16, 20),  # ring tip to pinky tip
]

def px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
mp_hands = mp.solutions.hands

# Extra edges to make it feel denser (between fingertips, across palm, etc.)
EXTRA_EDGES = [
    (4, 8), (8, 12), (12, 16), (16, 20),   # connect fingertip to fingertip
    (0, 5), (0, 9), (0, 13), (0, 17),      # wrist to knuckles
    (5, 9), (9, 13), (13, 17),             # across knuckles
]


mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=12,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

WIN = "Camera"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)   # allows resizing
cv2.resizeWindow(WIN, 1400, 900)          # pick any size you want


print(cap.isOpened())

while True:
    ret, frame = cap.read()
    if ret == False:
        break

    frame = cv2.flip(frame, 1)  # mirror (selfie view)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb = rgb.copy()  # forces contiguous, writable memory

    results = hands.process(rgb)
    face_results = face.process(rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # distances between fingers
            # distances between fingers
            for name, a, b in FINGER_GAPS:
                p1 = px(hand_landmarks.landmark[a], w, h)
                p2 = px(hand_landmarks.landmark[b], w, h)

                cv2.line(frame, p1, p2, (255, 255, 255), 2)  # <-- ADD HERE

                d = dist(p1, p2)  # pixels
                mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2  # midpoint

                cv2.putText(frame, f"{name}:{d:.0f}px", (mx, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if face_results.detections:
        h, w, _ = frame.shape
        for det in face_results.detections:
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # box metrics
            box_w = x2 - x1
            box_h = y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # example: "nose-ish" proxy = center of box to left edge
            nose_to_left = cx - x1
            nose_to_right = x2 - cx

            # draw a line from center to left edge
            cv2.line(frame, (cx, cy), (x1, cy), (255, 255, 255), 2)
            cv2.putText(frame, f"center->L:{nose_to_left}px", (x1, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # face box size label
            cv2.putText(frame, f"faceW:{box_w}px faceH:{box_h}px", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Camera", frame)

    # waitKey
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break




cv2.destroyAllWindows()
hands.close()
face.close()

cap.release()