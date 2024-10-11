import cv2
import numpy as np
from collections import deque

# Global variables for OpenCV
cap = cv2.VideoCapture(0)

# Setup canvas for drawing
paintWindow = np.zeros((471, 636, 3)) + 255
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]
colorIndex = 0
brush_size = 2

# Function to clear the canvas
def clear_canvas():
    global paintWindow
    paintWindow = np.zeros((471, 636, 3)) + 255

# Function to save the canvas as an image
def save_canvas():
    cv2.imwrite('drawing.png', paintWindow)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Example color detection (blue)
    Mask = cv2.inRange(hsv, (110, 50, 50), (130, 255, 255))
    cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center = None
    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Draw on the canvas based on the color index
        if center:
            if colorIndex == 0:
                bpoints[0].appendleft(center)
            elif colorIndex == 1:
                gpoints[0].appendleft(center)
            elif colorIndex == 2:
                rpoints[0].appendleft(center)
            elif colorIndex == 3:
                ypoints[0].appendleft(center)

    # Draw lines of all colors on the canvas
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], brush_size)

    # Show the frames in separate windows
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Paint Canvas", paintWindow)
    cv2.imshow("Mask View", Mask)

    # Key bindings
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Clear canvas
        clear_canvas()
    elif key == ord('s'):  # Save canvas
        save_canvas()
    elif key == ord('1'):  # Change color to blue
        colorIndex = 0
    elif key == ord('2'):  # Change color to green
        colorIndex = 1
    elif key == ord('3'):  # Change color to red
        colorIndex = 2
    elif key == ord('4'):  # Change color to yellow
        colorIndex = 3
    elif key == ord('+'):  # Increase brush size
        brush_size = min(10, brush_size + 1)
    elif key == ord('-'):  # Decrease brush size
        brush_size = max(1, brush_size - 1)

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
