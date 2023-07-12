import torch
import torch.nn as nn
import torch.optim as optim


class PoseChecking(nn.Module):
    def __init__(self):
        super(PoseChecking, self).__init__()
        self.fc1 = nn.Linear(34, 68)
        self.fc2 = nn.Linear(68, 34)
        self.fc3 = nn.Linear(34, 17)
        self.fc4 = nn.Linear(17, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


def train_model(model, train_data, train_labels, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()


def predict(model, test_data):
    outputs = model(test_data)
    predictions = (outputs > 0.5).squeeze().tolist()
    return predictions
