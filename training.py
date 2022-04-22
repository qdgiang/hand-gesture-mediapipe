import csv
import copy
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
from helper import *

cam_width = 960
cam_height = 540
min_detection_confidence  = 0.9
min_tracking_confidence = 0.8
keypoint_classifier_labels = ["fist", "reverse fist", "palm", "thumb_left", "thumb_right", "OK", "draw"]
def main():
    cap_device = 0
    cap_width = cam_width
    cap_height = cam_height
    use_brect = True
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    two_hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    history_length = 16
    point_history = deque(maxlen=history_length)
    mode = 0

    while True:
        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode)
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1) 
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                #hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                #if hand_sign_id == 6: 
                #if mode == 2:
                #    point_history.append(landmark_list[8]) # 人差指座標
                #else:
                #    point_history.append([0, 0])

                #finger_gesture_id = 0
                #point_history_len = len(pre_processed_point_history_list)
                #if point_history_len == (history_length * 2):
                #    finger_gesture_id = point_history_classifier(
                #        pre_processed_point_history_list)

                #finger_gesture_history.append(finger_gesture_id)
                #most_common_fg_id = Counter(
                #    finger_gesture_history).most_common()
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                """debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                    #point_history_classifier_labels[most_common_fg_id[0][0]],
                )"""
        #else:
            #point_history.append([0, 0])

        #debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'gen_data/landmark.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'gen_data/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

if __name__ == '__main__':
    main()
