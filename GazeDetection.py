import cv2
import dlib
import numpy as np
from collections import deque
import time

# Load the shape predictor model
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Initialize the dlib face detector
detector = dlib.get_frontal_face_detector()

# Initialize variables for pupil detection
pupil_history = []  # List to store the history of pupil positions
num_frames_accumulated = 15  # Number of frames to accumulate for stabilization

# Initialize the deque to store pupil positions for the last 10 seconds
position_history = deque()  # Assuming 10 frames per second


# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks (x, y) coordinates
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Compute the Euclidean distance between the horizontal eye landmark (x, y) coordinates
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect and locate pupils
def detect_pupils(image, shape):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the eye aspect ratio for both eyes
    left_eye_landmarks = np.array(shape[36:42])
    right_eye_landmarks = np.array(shape[42:48])
    left_ear = eye_aspect_ratio(left_eye_landmarks)
    right_ear = eye_aspect_ratio(right_eye_landmarks)

    # Define the thresholds for closed eyes
    ear_threshold = 0.2

    # Check if the eyes are closed or partially closed
    if left_ear < ear_threshold:
        left_eye_landmarks = np.concatenate([left_eye_landmarks, [shape[36]]])
    if right_ear < ear_threshold:
        right_eye_landmarks = np.concatenate([right_eye_landmarks, [shape[42]]])

    # Calculate the eye region of interest (ROI)
    left_eye_roi = cv2.boundingRect(left_eye_landmarks)
    right_eye_roi = cv2.boundingRect(right_eye_landmarks)

    # Get the eye regions from the grayscale image
    left_eye_region = gray[left_eye_roi[1]:left_eye_roi[1] + left_eye_roi[3],
                      left_eye_roi[0]:left_eye_roi[0] + left_eye_roi[2]]
    right_eye_region = gray[right_eye_roi[1]:right_eye_roi[1] + right_eye_roi[3],
                       right_eye_roi[0]:right_eye_roi[0] + right_eye_roi[2]]

    # Apply adaptive thresholding to enhance the pupil/iris contrast
    _, left_eye_thresh = cv2.threshold(left_eye_region, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, right_eye_thresh = cv2.threshold(right_eye_region, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find contours in the thresholded images
    left_eye_contours, _ = cv2.findContours(left_eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    right_eye_contours, _ = cv2.findContours(right_eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area in each eye region
    if len(left_eye_contours) > 0:
        left_eye_contour = max(left_eye_contours, key=cv2.contourArea)
        ((x, y), left_eye_radius) = cv2.minEnclosingCircle(left_eye_contour)
        left_eye_center = (int(x) + left_eye_roi[0], int(y) + left_eye_roi[1])
    else:
        left_eye_center = (left_eye_roi[0] + left_eye_roi[2] // 2, left_eye_roi[1] + left_eye_roi[3] // 2)

    if len(right_eye_contours) > 0:
        right_eye_contour = max(right_eye_contours, key=cv2.contourArea)
        ((x, y), right_eye_radius) = cv2.minEnclosingCircle(right_eye_contour)
        right_eye_center = (int(x) + right_eye_roi[0], int(y) + right_eye_roi[1])
    else:
        right_eye_center = (right_eye_roi[0] + right_eye_roi[2] // 2, right_eye_roi[1] + right_eye_roi[3] // 2)

    return left_eye_center, right_eye_center

# Function to classify the position of the pupil
def classify_pupil_position(avg_left_eye, avg_right_eye, displacement_threshold):
    # Get the left and right eye positions         print('')

    right_displacement = right_eye[0] - avg_right_eye[0]

    left_eye_landmarks = np.array(shape[36:42])
    right_eye_landmarks = np.array(shape[42:48])


    # Check if the average pupil position is closer to contour points 40 and 46 (left eye)
    if ((np.linalg.norm(right_eye_landmarks[3]-avg_right_eye))-(np.linalg.norm(avg_right_eye - right_eye_landmarks[0]))) > displacement_threshold:
        return "Right"
    # Check if the average pupil position is closer to contour points 37 and 43 (right eye)
    elif ((np.linalg.norm(avg_right_eye - right_eye_landmarks[0])) - (np.linalg.norm(right_eye_landmarks[3]-avg_right_eye))) > displacement_threshold:
        return "Left"
    else:
        return "Center"

def get_gaze_ratio(st, landmarks):
    eye_region = np.array([(landmarks.part(st).x, landmarks.part(st).y),
                           (landmarks.part(st+1).x, landmarks.part(st+1).y),
                           (landmarks.part(st+2).x, landmarks.part(st+2).y),
                           (landmarks.part(st+3).x, landmarks.part(st+3).y),
                           (landmarks.part(st+4).x, landmarks.part(st+4).y),
                           (landmarks.part(st+5).x, landmarks.part(st+5).y),
                           ], np.int32)
    
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    # cv2.imshow("mask", mask)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_eye = cv2.bitwise_and(gray, gray, mask=mask)
    # cv2.imshow("eye_region", gray_eye)

    # to get a frame around the eye region, from the landmark points:
    # left most point, is the one with the minx:
    min_x = np.min(eye_region[:, 0])
    # right most point for the eye frame is the one with the max x coordinate
    max_x = np.max(eye_region[:, 0])
    # # top-most pint for the eye frame, is the one with the min y coordinate
    min_y = np.min(eye_region[:, 1])
    # # bottom-most pnt for the frame is the one with max y
    max_y = np.max(eye_region[:, 1])
    # print(min_x)
    # print(min_y)
    only_eye = gray_eye[min_y:max_y, min_x:max_x]
    # pixels greater than 70 grayscale get a value 255: white
    _, threshold_eye = cv2.threshold(only_eye, 70, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', threshold_eye)
    # # in order to distinguish left and right gaze, i need to separate the left and right eye regions
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0: int(width/2)]
    left_side_white = np.count_nonzero(left_side_threshold)
    right_side_threshold = threshold_eye[0: max_y, int(width/2): width]
    right_side_white = np.count_nonzero(right_side_threshold)
    # print(left_side_white, right_side_white)
    # based on the differential black and white cells in each frame, left and right gaze can be determined
    if left_side_white > 0 and right_side_white > 0:  # to avoid div by 0 err
        gaze_ratio = left_side_white/right_side_white
        # print(f'gaze_ratio: {gaze_ratio}')
        return gaze_ratio

# Define a function to draw a plus symbol at a given position
def draw_plus_symbol(image, center_position, size, color):
    x, y = center_position
    half_size = size // 2  # Half the size of the plus symbol

    # Draw horizontal line of the plus symbol
    cv2.line(image, (x - half_size, y), (x + half_size, y), color, 2)

    # Draw vertical line of the plus symbol
    cv2.line(image, (x, y - half_size), (x, y + half_size), color, 2)


# Open a video capture object to capture video from a webcam (index 0)
cap = cv2.VideoCapture(0)
frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video capture

    if ret:
        # Detect faces in the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale
        faces = detector(gray)  # Use the face detector to detect faces in the grayscale frame

        for face in faces:
            # Determining facial landmarks for the detected face
            shape = predictor(gray, face)
            landmarks = predictor(gray, face)
            shape = [(p.x, p.y) for p in shape.parts()]
            # Detecting the pupils using the facial landmarks
            left_eye, right_eye = detect_pupils(frame, shape)

            # Accumulating pupil positions for stabilization
            pupil_history.append((left_eye, right_eye))


            if len(pupil_history) > num_frames_accumulated:
                pupil_history.pop(0)

            # Calculating the average pupil position
            avg_left_eye = np.mean(pupil_history, axis=0, dtype=int)[0]
            avg_right_eye = np.mean(pupil_history, axis=0, dtype=int)[1]
            # avg_right_eye = np.mean(pupil_history, axis=0, dtype=int)[1]

            # Detecting the Blinking state: if the eyes are closed, 
            # skip drawing the rectangle boxes and red pupil marks
            # Calculating the eye aspect ratio for both eyes
            left_eye_landmarks = np.array(shape[36:42])
            right_eye_landmarks = np.array(shape[42:48])
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)
            # Check if the eyes are closed
            ear_threshold = 0.2
            if left_ear < ear_threshold or right_ear < ear_threshold:
                # Eyes are closed, skip drawing the rectangle boxes and red pupil marks
                continue

            # Draw circles around the averaged pupil positions
            # cv2.circle(frame, tuple(avg_left_eye), 3, (255, 0, 0), -1)
            # cv2.circle(frame, tuple(avg_right_eye), 3, (255, 0, 0), -1)

            # Usage to draw a plus symbol
            draw_plus_symbol(frame, tuple(avg_left_eye), size=10, color=(255, 0, 255))
            draw_plus_symbol(frame, tuple(avg_right_eye), size=10, color=(255, 0, 255))

            # GAZE_DETECTION
            left_eye_gaze_ratio = get_gaze_ratio(36, landmarks)
            right_eye_gaze_ratio = get_gaze_ratio(42, landmarks)
            # print(left_eye_gaze_ratio, right_eye_gaze_ratio)
            indicator = np.zeros((500, 500, 3), np.uint8)
            if left_eye_gaze_ratio and right_eye_gaze_ratio:
                gaze_ratio = (left_eye_gaze_ratio + right_eye_gaze_ratio)/2
                # print gaze ratio for every 10 frames
                frame_counter += 1
                if frame_counter == 10:
                    print(gaze_ratio)
                    frame_counter = 0
                    
                if gaze_ratio <= 0.7:
                    indicator[:] = (0, 0, 255)
                    cv2.putText(frame, 'RIGHT', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                elif gaze_ratio <= 2.5:
                    indicator[:] = (0, 255, 0)
                    cv2.putText(frame, 'CENTER', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                else:
                    indicator[:] = (255, 0, 0)
                    cv2.putText(frame, 'LEFT', (50, 100),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # Classify the position of the pupil
            # displacement_threshold = 8  # Adjust this threshold as needed
            # position = classify_pupil_position(avg_left_eye, avg_right_eye, displacement_threshold)
            # if position != "Center":
            #     position_history.append(time.time())
            # else:
            #     position_history.clear()

            # # Check if the position has not been "Center" for a long time
            # attention_time_threshold = 5  # Time threshold in seconds
            # if len(position_history) > 0 and (time.time() - position_history[0]) > attention_time_threshold:
            #     # Display "Attention!" on the frame
            #     cv2.putText(frame, "Attention!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw the position text on the frame
            # cv2.putText(frame, f"Pupil Position: {position}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Define the text to be displayed
        text = f'FPS: {fps}'
        
        # Calculate the width and height of the text
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)        
        
        # Specify the position (x, y) for displaying FPS inside from the bottom-right corner
        x = frame.shape[1] - text_size[0] - 50
        y = frame.shape[0] - 50
        
        # display the fps on the frame
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
        
        # Display the resulting frame
        cv2.imshow('Pupil Position Detection', frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
