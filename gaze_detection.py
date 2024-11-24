import cv2
import numpy as np
import pyautogui
import dlib
from scipy.spatial import distance as dist
from yolo5face.get_model import get_model
from scipy.spatial import distance as dist
import time

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

class AdvancedGazeMovementControl:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing_factor = 1
        self.last_face_center = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0
        self.blink_start_time = None
        self.LEFT_EYE_AR_THRESH = 0.3
        self.RIGHT_EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.LEFT_COUNTER = 0
        self.RIGHT_COUNTER = 0
        self.LEFT_TOTAL = 0
        self.RIGHT_TOTAL = 0
        self.left_blink_start_time = None
        self.right_blink_start_time = None

    def detect_face_and_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            face_center = (rect.left() + rect.right()) // 2, (rect.top() + rect.bottom()) // 2
            
            return frame, face_center, left_ear, right_ear, left_eye, right_eye
        
        return frame, None, None, None, None, None
    def move_cursor(self, face_center, frame_shape):
        if self.last_face_center is None:
            self.last_face_center = face_center
            return

        delta_x = face_center[0] - self.last_face_center[0]
        delta_y = face_center[1] - self.last_face_center[1]

        self.last_face_center = face_center

        scale_factor = 5
        move_x = int(-delta_x * scale_factor)
        move_y = int(delta_y * scale_factor)

        current_x, current_y = pyautogui.position()
        new_x = current_x + move_x
        new_y = current_y + move_y

        new_x = max(0, min(new_x, self.screen_width - 1))
        new_y = max(0, min(new_y, self.screen_height - 1))

        pyautogui.moveTo(new_x, new_y)

    def process_frame(self, frame):
        frame, face_center, left_ear, right_ear, left_eye, right_eye = self.detect_face_and_eyes(frame)
        
        if face_center:
            cv2.circle(frame, face_center, 5, (255, 0, 0), -1)
            self.move_cursor(face_center, frame.shape)
            
            # Left eye (right-click)
            if left_ear < self.LEFT_EYE_AR_THRESH:
                if self.left_blink_start_time is None:
                    self.left_blink_start_time = time.time()
                self.LEFT_COUNTER += 1
            else:
                if self.LEFT_COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    blink_duration = time.time() - self.left_blink_start_time
                    if blink_duration >= 1:
                        self.LEFT_TOTAL += 1
                        pyautogui.click(button='right')
                        print(f"Left eye blink detected! Duration: {blink_duration:.2f}s. Total left blinks: {self.LEFT_TOTAL}")
                self.LEFT_COUNTER = 0
                self.left_blink_start_time = None
            
            # Right eye (left-click)
            if right_ear < self.RIGHT_EYE_AR_THRESH:
                if self.right_blink_start_time is None:
                    self.right_blink_start_time = time.time()
                self.RIGHT_COUNTER += 1
            else:
                if self.RIGHT_COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    blink_duration = time.time() - self.right_blink_start_time
                    if blink_duration >= 1:
                        self.RIGHT_TOTAL += 1
                        pyautogui.click()
                        print(f"Right eye blink detected! Duration: {blink_duration:.2f}s. Total right blinks: {self.RIGHT_TOTAL}")
                self.RIGHT_COUNTER = 0
                self.right_blink_start_time = None
            
            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Visualize both eyes
            if left_eye is not None and right_eye is not None:
                for (x, y) in left_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for (x, y) in right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        cv2.putText(frame, f"Left Blinks: {self.LEFT_TOTAL}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Right Blinks: {self.RIGHT_TOTAL}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

if __name__ == '__main__':
    gaze_movement_control = AdvancedGazeMovementControl()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = gaze_movement_control.process_frame(frame)
        cv2.imshow('Advanced Head Movement Control', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()