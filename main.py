import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Mediapipe hands
hands_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

# Gesture states
scroll_reference_y = None
click_down = False
last_screenshot_time = 0
scrolling = False  # flag to indicate continuous scroll

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            draw_utils.draw_landmarks(frame, hand_landmarks)
            lm = hand_landmarks.landmark
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]

            thumb_tip = pts[4]
            index_tip = pts[8]
            middle_tip = pts[12]
            ring_tip = pts[16]
            pinky_tip = pts[20]

            # Screenshot Gesture - all fingertips close together
            tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
            max_tip_dist = max(distance(t1, t2) for i, t1 in enumerate(tips) for t2 in tips[i+1:])
            if max_tip_dist < 50:  # all fingers tightly together
                now = time.time()
                if now - last_screenshot_time > 2:  # cooldown 2 sec
                    screenshot = pyautogui.screenshot()
                    screenshot.save(f"screenshot_{int(now)}.png")
                    last_screenshot_time = now
                    cv2.putText(frame, "📸 Screenshot Taken", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.imshow("Virtual Mouse", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Cursor - index finger
            cursor_x = screen_width / w * index_tip[0]
            cursor_y = screen_height / h * index_tip[1]
            pyautogui.moveTo(cursor_x, cursor_y)

            # Click - thumb + index pinch
            pinch_dist = distance(thumb_tip, index_tip)
            if pinch_dist < 40:
                cv2.line(frame, thumb_tip, index_tip, (0, 0, 255), 3)
                cv2.circle(frame, index_tip, 10, (0, 0, 255), -1)
                if not click_down:
                    pyautogui.click()
                    click_down = True
            else:
                click_down = False
                cv2.circle(frame, index_tip, 10, (0, 255, 255), -1)

            # Alt+Tab - thumb + pinky pinch
            thumb_pinky_dist = distance(thumb_tip, pinky_tip)
            if thumb_pinky_dist < 50:
                cv2.line(frame, thumb_tip, pinky_tip, (255, 0, 255), 3)
                pyautogui.hotkey('alt', 'tab')
                cv2.putText(frame, "Alt+Tab", (thumb_tip[0]-40, thumb_tip[1]-40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Scroll - index + middle joined & moved vertically
            index_middle_horizontal = abs(index_tip[0] - middle_tip[0]) < 40
            ring_folded = ring_tip[1] > pts[14][1]
            pinky_folded = pinky_tip[1] > pts[18][1]

            if index_middle_horizontal and ring_folded and pinky_folded:
                scrolling = True
                join_y = (index_tip[1] + middle_tip[1]) / 2

                # Change finger color to green during scroll
                cv2.circle(frame, index_tip, 10, (0, 255, 0), -1)
                cv2.circle(frame, middle_tip, 10, (0, 255, 0), -1)

                if scroll_reference_y is None:
                    scroll_reference_y = join_y

                delta = scroll_reference_y - join_y
                deadzone = 30

                if abs(delta) > deadzone:
                    # Continuous proportional scroll
                    scroll_amount = int(delta * 8)  # increase multiplier for faster scroll
                    pyautogui.scroll(scroll_amount)
                    scroll_reference_y = join_y

                    if scroll_amount > 0:
                        cv2.putText(frame, "Scroll Up", (index_tip[0]-60, index_tip[1]-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Scroll Down", (index_tip[0]-60, index_tip[1]-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                scrolling = False
                scroll_reference_y = None
                cv2.circle(frame, index_tip, 10, (0, 255, 255), -1)
                cv2.circle(frame, middle_tip, 10, (0, 255, 255), -1)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
