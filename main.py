import cv2
import numpy as np

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from webcam
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Define the region of interest for eye detection as the top half of the face
        roi_gray = gray[y:y + h // 2, x:x + w]  # Only top half
        roi_color = frame[y:y + h // 2, x:x + w]

        # Detect eyes within the top half of the face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Reduce the ROI to the top part of the eye (to exclude eyebrows)
            reduced_eye_region = int(eh * 0.65)  # Consider only the top 50% of the detected eye region
            start_y = int(eh* 0.3)
            end_y = int(eh * 0.68)
            eye_region = roi_gray[ey + start_y:ey + end_y, ex:ex + ew]
            eye_color = roi_color[ey + start_y:ey + end_y, ex:ex + ew]

            # Apply GaussianBlur to reduce noise
            eye_blur = cv2.GaussianBlur(eye_region, (7, 7), 0)

            # Apply adaptive thresholding to isolate the white part of the eye (sclera)
            eye_thresh = cv2.adaptiveThreshold(eye_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 11, 2)

            # Detect contours in the thresholded image
            contours, _ = cv2.findContours(eye_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Find the largest contour, which should correspond to the sclera
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)

                # Filter out contours that are too small or too large
                if 10 < contour_area < 110:  # Adjust these values based on your input size
                    # Fit an ellipse around the largest contour (approximate the shape of the eye)
                    if len(largest_contour) >= 5:  # FitEllipse needs at least 5 points
                        ellipse = cv2.fitEllipse(largest_contour)

                        # Draw the ellipse on the eye region
                        cv2.ellipse(eye_color, ellipse, (0, 255, 0), 2)

                        # The center of the ellipse can be considered as the eye's center
                        eye_center = (int(ellipse[0][0]), int(ellipse[0][1]))

                        # Draw a dot at the ellipse center
                        cv2.circle(eye_color, eye_center, 2, (255, 0, 0), 3)

                        # Optionally, calculate the angle of the ellipse (this gives the eye's rotation)
                        angle = ellipse[2]
                        cv2.putText(roi_color, f"Angle: {angle:.2f}", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Eye Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
vid.release()
cv2.destroyAllWindows()
