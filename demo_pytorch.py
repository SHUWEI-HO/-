import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
from find_camera import check_camera

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_num', default=0, type=int)
    parser.add_argument('--model_path', default='models/lstm_pose_model.pth')
    parser.add_argument('--scaler_path', default='scaler/scaler.pkl')
    parser.add_argument('--label_path', default='labels.txt')
    return parser.parse_args()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)  # 取得 batch size
        # 初始化 hidden state 和 cell state
        h_0 = torch.zeros(2, batch_size, 64).to(x.device)  # 2 是 LSTM 層數
        c_0 = torch.zeros(2, batch_size, 64).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))  # LSTM 推論
        out = out[:, -1, :]  # 取最後一個時間步的輸出
        out = self.fc(out)  # 通過全連接層
        return out

def compute_distances(landmarks):
    # 使用 18 個特徵配對
    pairs = [(11, 12), (13, 15), (14, 16), 
             (23, 25), (24, 26), (25, 27), 
             (26, 28), (15, 19), (16, 20), 
             (19, 21), (20, 22), (11, 23), 
             (12, 24), (27, 29), (28, 30), 
             (23, 27), (24, 28), (11, 24)]  # 總共 18 對

    distances = []
    reference_distance = np.linalg.norm(
        np.array([landmarks.landmark[11].x, landmarks.landmark[11].y]) -
        np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])
    )

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance / reference_distance)  # 正規化距離

    return distances

if __name__ == '__main__':
    args = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    input_dim = 18  # 輸入特徵數量
    hidden_dim = 64
    output_dim = len(open(args.label_path).readlines())  # 根據標籤數量設定輸出維度

    model = LSTMModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 載入 scaler 和標籤
    scaler = joblib.load(args.scaler_path)

    with open(args.label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # 開啟攝影機
    cap = check_camera(index=args.camera_num)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, 
                        smooth_landmarks=True, min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks
            distances = compute_distances(landmarks)

            if len(distances) != 18:
                raise ValueError(f"Expected 18 features, but got {len(distances)}.")

            distances = scaler.transform([distances])  # 標準化輸入
            distances = torch.tensor(distances, dtype=torch.float32).to(device)

            # 增加 batch 維度
            distances = distances.unsqueeze(0)  # shape: [1, 18]

            with torch.no_grad():
                output = model(distances)  # 推論
                _, predicted = torch.max(output.data, 1)
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()

            label = labels[predicted.item()]
            display_text = f"Pose: {label} ({confidence * 100:.2f}%)"
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255, 0, 0), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Demo', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # 按下 ESC 鍵退出
            break

    cap.release()
    cv2.destroyAllWindows()
