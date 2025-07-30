import cv2
import numpy as np
import mediapipe as mp

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas for drawing
canvas = np.zeros((480, 640, 3), np.uint8)
draw_color = (255, 0, 255)  # Default: Purple
xp, yp = 0, 0  # Previous points


def get_hand_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    lm_list = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
    return lm_list


while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    lm_list = get_hand_landmarks(frame)

    if lm_list:
        x1, y1 = lm_list[8][1:]   # Index tip
        x2, y2 = lm_list[12][1:]  # Middle tip

        # Check fingers up
        fingers_up = []
        fingers_up.append(lm_list[8][2] < lm_list[6][2])   # Index
        fingers_up.append(lm_list[12][2] < lm_list[10][2]) # Middle

        # Color selection mode
        if y1 < 50:
            if 50 < x1 < 150:
                draw_color = (255, 0, 255)
            elif 160 < x1 < 260:
                draw_color = (0, 255, 0)
            elif 270 < x1 < 370:
                draw_color = (0, 0, 255)
            elif 380 < x1 < 480:
                canvas = np.zeros((480, 640, 3), np.uint8)

        # Drawing mode
        if fingers_up[0] and not fingers_up[1]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(canvas, (xp, yp), (x1, y1), draw_color, 5)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    # Merge canvas and video
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, mask)
    frame = cv2.bitwise_or(frame, canvas)

    # Draw color boxes
    cv2.rectangle(frame, (50, 0), (150, 50), (255, 0, 255), -1)
    cv2.rectangle(frame, (160, 0), (260, 50), (0, 255, 0), -1)
    cv2.rectangle(frame, (270, 0), (370, 50), (0, 0, 255), -1)
    cv2.rectangle(frame, (380, 0), (480, 50), (0, 0, 0), -1)
    cv2.putText(frame, "Clear", (390, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Virtual Drawing", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break
cap.release()
cv2.destroyAllWindows()
