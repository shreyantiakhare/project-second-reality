import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

# Red color ranges (HSV)
lower_red1 = np.array([0, 150, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 80])
upper_red2 = np.array([179, 255, 255])

trail = deque(maxlen=64)       # newest position first

# Physics settings
GRAVITY = 1.0                  # pixels per frame^2
PREDICT_FRAMES = 20            # frames drawn into the future
FIT_WINDOW = 8                 # use the last 8 frames to estimate velocity

# Error measurement settings
ERROR_HORIZON = 5              # compare prediction vs reality after 5 frames
pending = deque()
frame_no = 0
errors = []
last_error = None

while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: Could not read frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 4, (0, 255, 255), -1)

    trail.appendleft(center)
    frame_no += 1

    # Compare old predictions with the real position
    while pending and pending[0][0] <= frame_no:
        target_frame, pred_x, pred_y = pending.popleft()
        if target_frame == frame_no and center is not None:
            last_error = float(np.hypot(center[0] - pred_x, center[1] - pred_y))
            if last_error < 150:
                errors.append(last_error)

    # Draw the actual trail (red)
    for i in range(1, len(trail)):
        if trail[i - 1] is None or trail[i] is None:
            continue
        cv2.line(frame, trail[i - 1], trail[i], (0, 0, 255), 2)

    # ---- PREDICTION (smoothed velocity) ----
    pts = [(-i, p) for i, p in enumerate(list(trail)[:FIT_WINDOW]) if p is not None]

    if len(pts) >= 4:
        t = np.array([q[0] for q in pts], dtype=float)
        xs = np.array([q[1][0] for q in pts], dtype=float)
        ys = np.array([q[1][1] for q in pts], dtype=float)

        # Straight-line fit: position = slope * t + intercept
        vx, x0 = np.polyfit(t, xs, 1)
        vy, y0 = np.polyfit(t, ys, 1)
        vy_start = vy

        # Draw predicted path (yellow dots)
        px, py = float(x0), float(y0)
        h, w = frame.shape[:2]
        for k in range(PREDICT_FRAMES):
            vy += GRAVITY
            px += vx
            py += vy
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            if k % 2 == 0:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 255), -1)

        # Save predicted position ERROR_HORIZON frames ahead
        ex, ey, evy = float(x0), float(y0), vy_start
        for _ in range(ERROR_HORIZON):
            evy += GRAVITY
            ex += vx
            ey += evy
        pending.append((frame_no + ERROR_HORIZON, ex, ey))

        cv2.putText(frame, "Prediction: Available", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Prediction: Waiting for data", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    status = "Ball: Tracked" if center else "Ball: Not found"
    cv2.putText(frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if last_error is not None and errors:
        avg = sum(errors) / len(errors)
        cv2.putText(frame, f"Prediction error: {last_error:.1f} px", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Average error: {avg:.1f} px", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if errors:
    print(f"GRAVITY = {GRAVITY}")
    print(f"Measurements: {len(errors)}")
    print(f"Average error: {sum(errors) / len(errors):.1f} px")
    print(f"Median error: {float(np.median(errors)):.1f} px")

cap.release()
cv2.destroyAllWindows()