import cv2
import mediapipe as mp
import numpy as np
import time
import os
import argparse
from find_camera import check_camera

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='dataset')
    parser.add_argument('--collect_mode', type=str, choices=['All', 'Upper', 'Lower', 'Hand'], default='Upper')
    parser.add_argument('--time', type=int, default=20)
    return parser.parse_args()

def compute_distances(landmarks, mode="All"):
    distances = []
    if mode == 'All':
        pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (23, 24), 
                 (23, 25), (25, 27), (24, 26), (26, 28), (28, 30), (27, 29)]
        reference_pair = (11, 23)
    elif mode == 'Upper':
        pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        reference_pair = (11, 12)
    elif mode == 'Lower':
        pairs = [(23, 24), (23, 25), (24, 26), (25, 27), (27, 29), (27, 31), 
                 (28, 30), (28, 32)]
        reference_pair = (23, 24)
    elif mode == 'Hand':
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), 
                 (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), 
                 (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), 
                 (4, 8), (8, 12), (12, 16), (16, 20)]
        reference_pair = (0, 9)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    p_ref1 = np.array([landmarks[reference_pair[0]].x, landmarks[reference_pair[0]].y, landmarks[reference_pair[0]].z])
    p_ref2 = np.array([landmarks[reference_pair[1]].x, landmarks[reference_pair[1]].y, landmarks[reference_pair[1]].z])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)

    for pair in pairs:
        p1 = np.array([landmarks[pair[0]].x, landmarks[pair[0]].y, landmarks[pair[0]].z])
        p2 = np.array([landmarks[pair[1]].x, landmarks[pair[1]].y, landmarks[pair[1]].z])
        distance = np.linalg.norm(p1 - p2) / reference_distance
        distances.append(distance)

    return distances

def initialize_model(mode):
    if mode in ['All', 'Upper', 'Lower']:
        mp_pose = mp.solutions.pose
        model = mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                             smooth_landmarks=True, min_detection_confidence=0.5, 
                             min_tracking_confidence=0.5)
        return model, mp_pose
    elif mode == 'Hand':
        mp_hands = mp.solutions.hands
        model = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                               min_detection_confidence=0.5, 
                               min_tracking_confidence=0.5)
        return model, mp_hands
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def process_frame(model, frame, mode):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if mode in ['All', 'Upper', 'Lower']:
        results = model.process(rgb_frame)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
    elif mode == 'Hand':
        results = model.process(rgb_frame)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark
    return None

def countdown(frame, seconds):
    for i in range(seconds, 0, -1):
        temp_frame = frame.copy()
        cv2.putText(temp_frame, str(i), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow("Data Collection", temp_frame)
        cv2.waitKey(1000)  # 每秒更新一次倒數

def main():
    args = get_parser()
    filename = args.file_name
    save_path = os.path.join(args.save_dir, filename)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cap = check_camera(0)
    model, mp_drawing = initialize_model(args.collect_mode)

    data_collection = []
    collecting = False
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not collecting:
            cv2.putText(frame, "Press SPACE to start data collection", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            elapsed_time = int(time.time() - start_time)
            remaining_time = args.time - elapsed_time
            cv2.putText(frame, f"Time left: {remaining_time} seconds", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if elapsed_time >= args.time:
                break

        landmarks = process_frame(model, frame, args.collect_mode)

        if landmarks and collecting:
            distances = compute_distances(landmarks, args.collect_mode)
            data_collection.append(distances)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1)

        if key == 32 and not collecting:  # 按下空白鍵開始收集
            countdown(frame, 3)  # 倒數3秒
            collecting = True
            start_time = time.time()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    np.save(save_path, np.array(data_collection))
    print(f"Data saved to {save_path}")

if __name__ == '__main__':
    main()
