import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Camera
cap = cv2.VideoCapture(0)

# Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Smooth mouse
prev_x, prev_y = 0, 0
smoothening = 5

# Canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# Brush settings
draw_color = (255, 0, 255)
brush_thickness = 5
eraser_thickness = 50

# Previous drawing points
xp, yp = 0, 0

# Click control
click_down = False

# Colors
colors = {
    "Purple": (255, 0, 255),
    "Blue": (255, 0, 0),
    "Green": (0, 255, 0),
    "Red": (0, 0, 255)
}

# Distance function
def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Toolbar
def draw_toolbar(frame):

    cv2.rectangle(frame, (10, 10), (110, 60), colors["Purple"], -1)
    cv2.rectangle(frame, (130, 10), (230, 60), colors["Blue"], -1)
    cv2.rectangle(frame, (250, 10), (350, 60), colors["Green"], -1)
    cv2.rectangle(frame, (370, 10), (470, 60), colors["Red"], -1)

    cv2.rectangle(frame, (490, 10), (620, 60), (50, 50, 50), -1)
    cv2.putText(frame, "ERASER", (510, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

# Main loop
while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    canvas = cv2.resize(canvas, (w, h))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    output = hands.process(rgb_frame)

    draw_toolbar(frame)

    if output.multi_hand_landmarks:

        for hand in output.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand.landmark

            points = []

            for landmark in landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                points.append((x, y))

            # Finger points
            thumb_tip = points[4]
            index_tip = points[8]
            middle_tip = points[12]
            ring_tip = points[16]
            pinky_tip = points[20]

            # Finger states
            index_up = index_tip[1] < points[6][1]
            middle_up = middle_tip[1] < points[10][1]

            # Mouse movement
            screen_x = np.interp(
                index_tip[0],
                (0, w),
                (0, screen_width)
            )

            screen_y = np.interp(
                index_tip[1],
                (0, h),
                (0, screen_height)
            )

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # Draw fingertips
            cv2.circle(frame, index_tip, 10, (0, 255, 255), -1)
            cv2.circle(frame, thumb_tip, 10, (255, 0, 255), -1)

            # Click gesture
            click_distance = distance(thumb_tip, index_tip)

            if click_distance < 40:

                cv2.line(frame,
                         thumb_tip,
                         index_tip,
                         (0, 0, 255),
                         3)

                if not click_down:
                    pyautogui.click()
                    click_down = True

            else:
                click_down = False

            # Selection mode
            if index_up and middle_up:

                xp, yp = 0, 0

                cv2.putText(frame,
                            "SELECT MODE",
                            (850, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            3)

                x1, y1 = index_tip

                if y1 < 80:

                    if 10 < x1 < 110:
                        draw_color = colors["Purple"]

                    elif 130 < x1 < 230:
                        draw_color = colors["Blue"]

                    elif 250 < x1 < 350:
                        draw_color = colors["Green"]

                    elif 370 < x1 < 470:
                        draw_color = colors["Red"]

                    elif 490 < x1 < 620:
                        draw_color = (0, 0, 0)

            # Draw mode
            elif index_up and not middle_up:

                cv2.putText(frame,
                            "DRAW MODE",
                            (850, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3)

                cv2.circle(frame,
                           index_tip,
                           15,
                           draw_color,
                           -1)

                if xp == 0 and yp == 0:
                    xp, yp = index_tip

                # Brush size
                brush_distance = distance(
                    thumb_tip,
                    index_tip
                )

                brush_thickness = int(
                    np.interp(
                        brush_distance,
                        [20, 200],
                        [2, 30]
                    )
                )

                # Eraser
                if draw_color == (0, 0, 0):

                    cv2.line(
                        canvas,
                        (xp, yp),
                        index_tip,
                        draw_color,
                        eraser_thickness
                    )

                else:

                    cv2.line(
                        canvas,
                        (xp, yp),
                        index_tip,
                        draw_color,
                        brush_thickness
                    )

                xp, yp = index_tip

            else:
                xp, yp = 0, 0

            # Clear canvas
            fist = (
                index_tip[1] > points[6][1] and
                middle_tip[1] > points[10][1] and
                ring_tip[1] > points[14][1] and
                pinky_tip[1] > points[18][1]
            )

            if fist:

                canvas = np.zeros((h, w, 3), dtype=np.uint8)

                cv2.putText(frame,
                            "CANVAS CLEARED",
                            (350, 300),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            4)

    # Merge canvas
    gray_canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, inverse = cv2.threshold(
        gray_canvas,
        50,
        255,
        cv2.THRESH_BINARY_INV
    )

    inverse = cv2.cvtColor(
        inverse,
        cv2.COLOR_GRAY2BGR
    )

    frame = cv2.bitwise_and(frame, inverse)

    frame = cv2.bitwise_or(frame, canvas)

    # Instructions
    cv2.putText(frame,
                "Index = Move Mouse",
                (20, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                "Thumb + Index = Click",
                (20, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                "1 Finger = Draw",
                (20, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                "2 Fingers = Select",
                (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    # Show output
    cv2.imshow(
        "AI Virtual Mouse + Air Canvas",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()