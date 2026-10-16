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

GRAVITY = 1.0
PREDICT_FRAMES = 20
FIT_WINDOW = 8
ERROR_HORIZON = 5
HEADER_H = 50
BAR_H = 110

# Colors (BGR)
GREEN = (120, 255, 120)
YELLOW = (0, 220, 255)
RED = (80, 80, 255)
CYAN = (255, 230, 80)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)
DARK = (22, 18, 18)
FONT = cv2.FONT_HERSHEY_DUPLEX

pending = deque()
frame_no = 0
errors = []
last_error = None
last_cmp = None
last_cmp_frame = -100
writer = None
grid = None


def err_color(e):
    """Green = small error, yellow = medium, red = large."""
    if e < 20:
        return GREEN
    if e < 50:
        return YELLOW
    return RED


def brackets(img, color, size=22, pad=8, thick=2):
    """Draw camera-style corner brackets."""
    h, w = img.shape[:2]
    for (x, y, dx, dy) in [(pad, pad, 1, 1), (w - pad, pad, -1, 1),
                           (pad, h - pad, 1, -1), (w - pad, h - pad, -1, -1)]:
        cv2.line(img, (x, y), (x + dx * size, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * size), color, thick, cv2.LINE_AA)


def stat(img, x, label, value, color):
    """One stat block: small label on top, big number below."""
    cv2.putText(img, label, (x, 38), FONT, 0.5, GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, value, (x, 82), FONT, 1.1, color, 2, cv2.LINE_AA)


while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: Could not read frame")
        break

    h, w = frame.shape[:2]

    if writer is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps < 5 or fps > 120:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("demo_pro.mp4", fourcc, fps,
                                 (w * 2, HEADER_H + h + BAR_H))
        # Pre-build the grid background for the right panel
        grid = np.full((h, w, 3), DARK, dtype=np.uint8)
        for gx in range(0, w, 40):
            cv2.line(grid, (gx, 0), (gx, h), (45, 40, 40), 1)
        for gy in range(0, h, 40):
            cv2.line(grid, (0, gy), (w, gy), (45, 40, 40), 1)

    # ---------- Detection ----------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    radius_px = 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 10:
            center = (int(x), int(y))
            radius_px = int(radius)

    trail.appendleft(center)
    frame_no += 1

    # ---------- Compare old predictions with reality ----------
    while pending and pending[0][0] <= frame_no:
        target_frame, pred_x, pred_y = pending.popleft()
        if target_frame == frame_no and center is not None:
            err = float(np.hypot(center[0] - pred_x, center[1] - pred_y))
            if err < 150:
                errors.append(err)
                last_error = err
                last_cmp = (pred_x, pred_y, center[0], center[1], err)
                last_cmp_frame = frame_no

    # ---------- LEFT PANEL: real video ----------
    left = frame.copy()
    if center is not None:
        cv2.circle(left, center, radius_px, GREEN, 2, cv2.LINE_AA)
        cv2.circle(left, center, 4, YELLOW, -1, cv2.LINE_AA)
        cv2.putText(left, "BALL", (center[0] - 20, max(20, center[1] - radius_px - 10)),
                    FONT, 0.55, GREEN, 1, cv2.LINE_AA)
    brackets(left, WHITE)
    cv2.putText(left, "REALITY", (20, 40), FONT, 0.8, WHITE, 1, cv2.LINE_AA)
    cv2.circle(left, (w - 28, 36), 6, GREEN if center else RED, -1, cv2.LINE_AA)

    # ---------- RIGHT PANEL: simulation ----------
    right = grid.copy()
    brackets(right, CYAN)
    cv2.putText(right, "SIMULATION", (20, 40), FONT, 0.8, CYAN, 1, cv2.LINE_AA)

    # Fading actual trail: newest = bright, oldest = dark
    pts_list = list(trail)
    n = len(pts_list)
    for i in range(1, n):
        if pts_list[i - 1] is None or pts_list[i] is None:
            continue
        f = 1.0 - i / float(n)
        col = (int(60 * f), int(60 * f), int(80 + 175 * f))
        cv2.line(right, pts_list[i - 1], pts_list[i], col,
                 max(1, int(3 * f)), cv2.LINE_AA)
    if center is not None:
        cv2.circle(right, center, 9, GREEN, -1, cv2.LINE_AA)
        cv2.circle(right, center, 13, GREEN, 1, cv2.LINE_AA)

    # Prediction (smoothed velocity)
    pts = [(-i, p) for i, p in enumerate(pts_list[:FIT_WINDOW]) if p is not None]
    pred_ready = False
    if len(pts) >= 4:
        t = np.array([q[0] for q in pts], dtype=float)
        xs = np.array([q[1][0] for q in pts], dtype=float)
        ys = np.array([q[1][1] for q in pts], dtype=float)
        vx, x0 = np.polyfit(t, xs, 1)
        vy, y0 = np.polyfit(t, ys, 1)
        vy_start = vy

        px, py = float(x0), float(y0)
        for k in range(PREDICT_FRAMES):
            vy += GRAVITY
            px += vx
            py += vy
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            size = max(1, 4 - k // 6)
            cv2.circle(right, (int(px), int(py)), size, YELLOW, -1, cv2.LINE_AA)

        ex, ey, evy = float(x0), float(y0), vy_start
        for _ in range(ERROR_HORIZON):
            evy += GRAVITY
            ex += vx
            ey += evy
        pending.append((frame_no + ERROR_HORIZON, ex, ey))
        pred_ready = True

    # Latest predicted vs real comparison, with colored error label
    if last_cmp is not None and frame_no - last_cmp_frame < 15:
        pxx, pyy, rx, ry, e = last_cmp
        col = err_color(e)
        p1 = (int(pxx), int(pyy))
        p2 = (int(rx), int(ry))
        cv2.circle(right, p1, 9, CYAN, 2, cv2.LINE_AA)
        cv2.circle(right, p2, 9, GREEN, 2, cv2.LINE_AA)
        cv2.line(right, p1, p2, col, 2, cv2.LINE_AA)
        mid = ((p1[0] + p2[0]) // 2 + 8, (p1[1] + p2[1]) // 2 - 8)
        cv2.putText(right, f"{e:.1f} px", mid, FONT, 0.6, col, 1, cv2.LINE_AA)

    # Legend
    cv2.putText(right, "red: actual   yellow: predicted", (20, h - 40),
                FONT, 0.45, GRAY, 1, cv2.LINE_AA)
    cv2.putText(right, "cyan ring: predicted   green ring: real", (20, h - 20),
                FONT, 0.45, GRAY, 1, cv2.LINE_AA)

    # ---------- HEADER ----------
    header = np.full((HEADER_H, w * 2, 3), (35, 28, 28), dtype=np.uint8)
    cv2.putText(header, "PROJECT SECOND REALITY", (20, 34), FONT, 0.9,
                WHITE, 1, cv2.LINE_AA)
    cv2.putText(header, "Reality vs Simulation Engine", (w * 2 - 330, 32), FONT,
                0.6, CYAN, 1, cv2.LINE_AA)

    # ---------- BOTTOM BAR ----------
    bar = np.full((BAR_H, w * 2, 3), (28, 22, 22), dtype=np.uint8)
    cv2.line(bar, (0, 0), (w * 2, 0), (70, 60, 60), 1)
    if errors:
        med = float(np.median(errors))
        avg = sum(errors) / len(errors)
        cur = last_error if last_error is not None else 0.0
        stat(bar, 20, "PREDICTION ERROR", f"{cur:.1f} px", err_color(cur))
        stat(bar, 340, "AVERAGE", f"{avg:.1f} px", err_color(avg))
        stat(bar, 600, "MEDIAN", f"{med:.1f} px", err_color(med))
        stat(bar, 860, "MEASUREMENTS", f"{len(errors)}", WHITE)
    else:
        stat(bar, 20, "PREDICTION ERROR", "measuring...", GRAY)
    obj = "Ball" if center is not None else "Not found"
    cv2.putText(bar, f"OBJECT: {obj}", (w * 2 - 330, 38), FONT, 0.55,
                GREEN if center else RED, 1, cv2.LINE_AA)
    cv2.putText(bar, f"PREDICTION: {'available' if pred_ready else 'waiting'}",
                (w * 2 - 330, 68), FONT, 0.55,
                YELLOW if pred_ready else GRAY, 1, cv2.LINE_AA)

    combined = np.vstack([header, np.hstack([left, right]), bar])

    writer.write(combined)
    cv2.imshow("Second Reality", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if errors:
    print(f"GRAVITY = {GRAVITY}")
    print(f"Measurements: {len(errors)}")
    print(f"Average error: {sum(errors) / len(errors):.1f} px")
    print(f"Median error: {float(np.median(errors)):.1f} px")

if writer is not None:
    writer.release()
cap.release()
cv2.destroyAllWindows()
print("Video saved as demo_pro.mp4")