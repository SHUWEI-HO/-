import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
import time  # 新增此行
from find_camera import check_camera 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_num', default=0)
    parser.add_argument('--model_path', default='models/svm_model.pkl')
    parser.add_argument('--scaler_path', default='scaler/scaler.pkl')
    parser.add_argument('--label_path', default='labels.txt')
    return parser.parse_args()

def compute_distances(landmarks):
    pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
             (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
             (28, 30), (27, 29), (15, 19), (15, 17), (19, 21),
             (16, 20), (16, 18), (20, 22)]

    distances = []
    reference_distance = np.linalg.norm(
        np.array([landmarks.landmark[11].x, landmarks.landmark[11].y]) -
        np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])
    )

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance / reference_distance)

    return distances

if __name__ == '__main__':
    args = get_parser()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        smooth_landmarks=True, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    clf = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)

    with open(args.label_path, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    cap = check_camera(index=args.camera_num)

    # 初始化計時器和顯示變數
    last_prediction_time = time.time()  
    last_label = "Initializing..."  # 初始顯示的文字
    last_confidence = 0.0  # 初始信心值

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        # 每五秒進行一次預測
        if time.time() - last_prediction_time >= 5:
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks
                distances = compute_distances(landmarks)
                distances = scaler.transform([distances])

                prediction = clf.predict(distances)
                confidence = np.max(clf.predict_proba(distances))

                # 更新顯示的標籤和信心值
                last_label = labels[prediction[0]]
                last_confidence = confidence

            # 更新最後一次預測的時間
            last_prediction_time = time.time()

        # 繪製姿勢關鍵點
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 長久顯示最新的預測結果
        display_text = f"Pose: {last_label} ({last_confidence*100:.2f}%)"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Demo', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按下 ESC 鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()
