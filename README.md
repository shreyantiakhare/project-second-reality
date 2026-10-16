# Project Second Reality
**Reality vs Simulation Engine**: a webcam turns into a small physics lab.

A webcam tracks a red ball, a simple physics model predicts where it will go,
and the system measures how wrong the prediction was.

## What it does
- Detects the ball by colour (HSV, OpenCV)
- Tracks its position and draws a trail
- Predicts the path using a straight-line fit of recent positions plus gravity
- Compares the prediction (5 frames ahead) with the real position
- Shows the error in pixels, live, side by side: Reality | Simulation

## Results (measured on my laptop webcam)
| Version | Measurements | Median error |
|---|---|---|
| Two-point velocity | 64 | 84.3 px |
| Smoothed velocity (line fit) | 63 | 48.9 px |
| Smoothed + final demo (demo_pro.py) | 190 | 20.2 px |

Setup: GRAVITY = 1.0 px/frame^2, 5-frame horizon, one red ball, plain wall.

## Limitations (honest notes)
- Runs were done by hand, so ball motion differed between runs.
  The numbers are indicative, not a controlled benchmark.
- Gravity is in pixels/frame^2 and depends on camera distance and FPS.
  My automatic calibration gave inconsistent values (about 2.6 to 11.6),
  so I picked GRAVITY by comparing prediction error instead.
- Error rises when the hand hides part of the ball (centre jumps).
- Friction and camera movement are not modelled yet.

## Run it
pip install opencv-python numpy
python demo_pro.py
Press `q` to quit. A video `demo_pro.mp4` is saved.

## Tech
Python, OpenCV, NumPy

## Next steps
- Automatic error-cause labels (tracking / physics / camera)
- Kalman filter for tracking
- Friction and air-drag model
