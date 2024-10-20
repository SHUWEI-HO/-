import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
from find_camera import check_camera 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_num', default=0)
    parser.add_argument('--model_path', default='models\\svm_model.pkl')
    parser.add_argument('--scaler_path', default='scaler\\scaler.pkl')
    parser.add_argument('--label_path', default='labels.txt')
    parser.add_argument('--collect_mode', type=str, choices=['All', 'Upper', 'Lower', 'Hand'], default='Upper')
    return parser.parse_args()

def compute_distances(landmarks, mode='Upper'):
    distances = []
    if mode == 'All':
        pairs = [(11, 12), (11, 13), (13, 15),
                 (12, 14), (14, 16),
                 (23, 24), (23, 25), (25, 27),
                 (24, 26), (26, 28),
                 (28, 30), (27, 29)]
        reference_pair = (11, 23)
    elif mode == 'Upper':
        pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
        reference_pair = (11, 12)
    elif mode == 'Lower':
        pairs = [(23, 24), (23, 25), (24, 26), (25, 27),
                 (27, 29), (27, 31), (28, 30), (28, 32)]
        reference_pair = (23, 24)
    elif mode == 'Hand':
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
                 (0, 5), (0, 6), (0, 7), (0, 8),
                 (0, 9), (0, 10), (0, 11), (0, 12),
                 (0, 13), (0, 14), (0, 15), (0, 16),
                 (0, 17), (0, 18), (0, 19), (0, 20),
                 (4, 8), (8, 12), (12, 16), (16, 20)]
        reference_pair = (0, 9)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    p_ref1 = np.array([landmarks[reference_pair[0]].x, landmarks[reference_pair[0]].y])
    p_ref2 = np.array([landmarks[reference_pair[1]].x, landmarks[reference_pair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)

    for pair in pairs:
        p1 = np.array([landmarks[pair[0]].x, landmarks[pair[0]].y])
        p2 = np.array([landmarks[pair[1]].x, landmarks[pair[1]].y])
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance / reference_distance)  # Normalize distance

    return distances

def initialize_model(mode):
    if mode in ['All', 'Upper', 'Lower']:
        mp_pose = mp.solutions.pose
        model = mp_pose.Pose(static_image_mode=False,
                             model_complexity=2,
                             smooth_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)
        return model, mp_pose
    elif mode == 'Hand':
        mp_hands = mp.solutions.hands
        model = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
        return model, mp_hands
    else:
        raise ValueError(f"Unsupported mode: {mode}")

if __name__ == '__main__':
    args = get_parser()

    # Initialize the appropriate model based on the collect_mode
    model, mp_module = initialize_model(args.collect_mode)
    model_type = 'pose' if args.collect_mode in ['All', 'Upper', 'Lower'] else 'hands'

    # Load the classifier and scaler
    clf = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)

    # Load labels
    label_file = args.label_path
    with open(label_file, 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]

    # Initialize the camera
    cap = check_camera(index=int(args.camera_num))

    # Prepare the drawing utility
    mp_drawing = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if model_type == 'pose':
            results = model.process(rgb_frame)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                distances = compute_distances(landmarks, args.collect_mode)
                distances = scaler.transform([distances])

                prediction = clf.predict(distances)
                confidence = np.max(clf.predict_proba(distances))

                label = labels[int(prediction[0])]
                display_text = f"Pose: {label} ({confidence*100:.2f}%)"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_module.POSE_CONNECTIONS)
        elif model_type == 'hands':
            results = model.process(rgb_frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    distances = compute_distances(landmarks, args.collect_mode)
                    distances = scaler.transform([distances])

                    prediction = clf.predict(distances)
                    confidence = np.max(clf.predict_proba(distances))

                    label = labels[int(prediction[0])]
                    display_text = f"Hand Gesture: {label} ({confidence*100:.2f}%)"
                    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_module.HAND_CONNECTIONS)

        # Display the video frame
        cv2.imshow('Demo', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
