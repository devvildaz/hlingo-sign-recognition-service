import cv2
import uuid
import os
import mediapipe as mp
import itertools
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class MediaPipeProcessor():
    _TMP = './tmp'
    _mp_hands = mp.solutions.hands
    _mp_pose = mp.solutions.pose
    _NUM_SAMPLES = int(os.environ['NUM_SAMPLES'])
    _NUM_STEPS = int(os.environ['NUM_STEPS'])
    _NUM_FEATURES = int(os.environ['NUM_FEATURES'])
    _FEATURES_PER_MODEL = int(_NUM_FEATURES / 3)
    _RESOLUTION = (int(os.environ['VIDEO_HEIGHT']), int(os.environ['VIDEO_WIDTH']))

    def __init__(self):
        if not os.path.exists(self._TMP):
            os.mkdir(self._TMP)

    def get_coordinates(self, request_video):
        video_path = os.path.join(self._TMP, f'{str(uuid.uuid4())}.mp4')
        self._save_tmp_video(video_path, request_video)
        coordinates = self._process_video(video_path)
        self._delete_tmp_video(video_path)
        return coordinates

    def _process_video(self, video_path):
        landmark_coords = np.zeros((self._NUM_SAMPLES, self._NUM_STEPS, self._NUM_FEATURES))

        with (self._mp_hands.Hands(min_detection_confidence=0.8) as hands,
            self._mp_pose.Pose(min_detection_confidence=0.8) as pose):
            
            cap = cv2.VideoCapture(video_path)
            sample = self._NUM_SAMPLES - 1
            step_iter = itertools.count()

            while cap.isOpened():
                step = next(step_iter)
                if step >= self._NUM_STEPS:
                    break

                ret, frame = cap.read()
                if ret == False:
                    break

                frame = cv2.resize(frame, self._RESOLUTION)
                image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
                hands_results = hands.process(image)
                pose_results = pose.process(image)

                landmark_coords = self._get_landmark_coordinates(landmark_coords,
                                                            hands_results,
                                                            pose_results,
                                                            sample,
                                                            step)
            cap.release()
        return landmark_coords

    def _get_landmark_coordinates(self, arr, hand_results, pose_results, sample, step):

        if hand_results.multi_hand_landmarks is not None:
            for k, landmarks in enumerate(hand_results.multi_hand_landmarks):
                if hand_results.multi_handedness[k].classification[0].label == 'Right':
                    for i, landmark in enumerate(hand_results.multi_hand_landmarks[k].landmark):
                        arr[sample][step][2 * i] = landmark.x
                        arr[sample][step][(2 * i) + 1] = landmark.y

                elif hand_results.multi_handedness[k].classification[0].label == 'Left':
                    for i, landmark in enumerate(hand_results.multi_hand_landmarks[k].landmark):
                        arr[sample][step][self._FEATURES_PER_MODEL + (2 * i)] = landmark.x
                        arr[sample][step][self._FEATURES_PER_MODEL + (2 * i) + 1] = landmark.y
    
        if pose_results.pose_landmarks is not None:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark[:21]):
                arr[sample][step][2 * self._FEATURES_PER_MODEL + (2 * i)] = landmark.x
                arr[sample][step][2 * self._FEATURES_PER_MODEL + (2 * i) + 1] = landmark.y
        
        return arr

    def _save_tmp_video(self, video_path, request_video):
        tmp_video = open(video_path, 'wb')
        tmp_video.write(request_video)
        tmp_video.close()

    def _delete_tmp_video(self, video_path):
        os.remove(video_path)

    def _create_response(self, coordinates):
        response = {}
        response['coordinates'] = self._format_coordinates(coordinates)
        return response

    def _format_coordinates(coordinates):
        return coordinates.tolist()