import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class CustomizedDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            row = self.transform(row)
        else:
            row = torch.FloatTensor(row)
        return row, label


def load():
    df = pd.read_csv("Train1.0.csv", sep=',', index_col=0)

    msk = np.random.rand(len(df)) <= 0.75
    train = df[msk]
    test = df[~msk]

    train_labels = train['churn'].values
    test_labels = test['churn'].values

    train_data = train.drop('churn', axis=1)
    test_data = test.drop('churn', axis=1)

    min = train_data.min()
    max = train_data.max()

    train_data = (train_data - min) / (max - min)
    test_data = (test_data - min) / (max - min)

    train_data = train_data.values
    test_data = test_data.values

    train_set = CustomizedDataset(train_data, train_labels, transform=None)
    test_set = CustomizedDataset(test_data, test_labels, transform=None)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=True)
    return train_loader, test_loader


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        # self.bn = nn.BatchNorm1d(12)
        self.fc1 = nn.Linear(12, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 2)

    def forward(self, x):
        # x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(train_loader: DataLoader):
    net = FCNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150, 300, 500, 700], gamma=0.8)

    print("Start Training...")
    for epoch in range(1000):
        loss_all = 0.0
        n = len(train_loader)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()

        print('[Epoch %d] loss: %.3f' % (epoch + 1, loss_all / n))
        scheduler.step()

    print("Done Training!")
    return net


def test(test_loader: DataLoader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            rows, labels = data
            outputs = net(rows)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # print(predicted)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test set: %.3f' % (correct / total))


if __name__ == '__main__':
    train_loader, test_loader = load()
    net = train(train_loader)
    test(test_loader, net)
    test(train_loader, net)
