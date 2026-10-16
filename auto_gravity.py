import cv2
import numpy as np

cap = cv2.VideoCapture(0)

lower_red1 = np.array([0, 150, 80])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 80])
upper_red2 = np.array([179, 255, 255])

positions = []      # y positions of the current throw
missing = 0         # frames in a row where the ball was not found
estimates = []      # gravity estimate from each throw

print("Throw the ball up (stay inside the frame) and catch it. Press q to quit.")

def finish_throw(ys):
    """Fit a parabola to one throw and return gravity (px/frame^2) or None."""
    if len(ys) < 12:
        return None
    t = np.arange(len(ys))
    a, b, c = np.polyfit(t, ys, 2)
    g = 2 * a
    # A real throw: the ball is highest (smallest y) in the middle
    peak = int(np.argmin(ys))
    if g > 0 and 3 < peak < len(ys) - 3:
        return g
    return None

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

    if center is not None:
        missing = 0
        positions.append(center[1])
        if len(positions) > 60:
            positions.pop(0)
    else:
        missing += 1

    # Ball disappeared for 5 frames, or the recording got long: finish the throw
    if (missing == 5 and positions) or len(positions) >= 40:
        g = finish_throw(positions)
        if g is not None:
            estimates.append(g)
            print(f"Throw {len(estimates)}: gravity = {g:.3f} px/frame^2")
        positions = []

    if estimates:
        text = f"Throws: {len(estimates)}  Median gravity: {np.median(estimates):.3f}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Auto gravity", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if estimates:
    print("FINAL median gravity:", round(float(np.median(estimates)), 3))

cap.release()
cv2.destroyAllWindows()