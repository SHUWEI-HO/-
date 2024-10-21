import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output', type=str, default='models')
    return parser.parse_args()

class PoseDataset(Dataset):
    def __init__(self, dataset_path, max_time_steps):
        self.X, self.y = self.load_data(dataset_path, max_time_steps)

    def load_data(self, dataset_path, max_time_steps):
        X, y = [], []

        class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        labels = {name: idx for idx, name in enumerate(class_dirs)}

        for class_name, label in labels.items():
            class_dir = os.path.join(dataset_path, class_name)
            for file in os.listdir(class_dir):
                if file.endswith('.npy'):
                    data = np.load(os.path.join(class_dir, file))
                    if len(data.shape) != 2:
                        print(f"跳過非2D資料：{file}")
                        continue

                    if data.shape[0] > max_time_steps:
                        data = data[:max_time_steps, :]

                    pad_width = max_time_steps - data.shape[0]
                    padded_data = np.pad(data, ((0, pad_width), (0, 0)), mode='constant')
                    X.append(padded_data)
                    y.append(label)

        X = np.array(X)
        y = np.array(y)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 64).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            batch_loss = criterion(outputs, y_batch)
            loss += batch_loss.item() * X_batch.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    loss /= len(dataloader.dataset)
    accuracy = 100 * correct / total
    return loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    args = get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_time_steps = 100
    dataset = PoseDataset(args.dataset, max_time_steps)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = dataset[0][0].shape[1]
    output_dim = len(set(dataset.y.numpy()))

    model = LSTMModel(input_dim, 64, output_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs)

    os.makedirs(args.output, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output, 'lstm_pose_model.pth'))
    print("Model saved successfully!")
