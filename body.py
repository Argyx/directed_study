import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

class FullBodyVisualizer:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_solutions = mp.solutions

        # Initialize Pose Landmarker
        self.pose_landmarker = self.create_pose_landmarker()

        # Initialize Hand Landmarker with a callback function
        self.hand_landmarker = self.create_hand_landmarker()

    def create_pose_landmarker(self):
        base_options = mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task')
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def create_hand_landmarker(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,  # Change to IMAGE mode
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3)
        return mp.tasks.vision.HandLandmarker.create_from_options(options)


        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            result_callback=hand_landmark_callback)
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        # Convert frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Detect pose and hands
        pose_result = self.pose_landmarker.detect(mp_image)
        hand_result = self.hand_landmarker.detect(mp_image)

        # Draw pose and hand landmarks
        annotated_image = self.draw_pose_landmarks(frame, pose_result)
        annotated_image = self.draw_hand_landmarks(annotated_image, hand_result)

        return annotated_image

    def draw_pose_landmarks(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for pose_landmarks in pose_landmarks_list:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            self.mp_solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                self.mp_solutions.pose.POSE_CONNECTIONS,
                self.mp_solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def draw_hand_landmarks(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        for hand_landmarks in hand_landmarks_list:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)  # Corrected typo here
                for landmark in hand_landmarks
            ])
            self.mp_solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                self.mp_solutions.hands.HAND_CONNECTIONS,
                self.mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
                self.mp_solutions.drawing_styles.get_default_hand_connections_style())
        return annotated_image


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Initialize FullBodyVisualizer
    visualizer = FullBodyVisualizer()

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Mirror frame

        # Process frame for pose and hand landmarks
        annotated_image = visualizer.process_frame(frame)

        # Display image
        cv2.imshow('Full Body Visualizer', annotated_image)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()