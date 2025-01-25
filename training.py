# -*- coding: utf-8 -*-
"""AO.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RoQoyL36U3avwg3sX2OdsTj9hb7iF-qH
"""
import subprocess
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import kagglehub
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    import kagglehub

    print(torch.cuda.is_available())

    # Download latest version
    path = kagglehub.dataset_download(
        "kritikseth/fruit-and-vegetable-image-recognition"
    )

    print("Path to dataset files:", path)
    training_path = path + "/train/"
    test_path = path + "/test/"
    import os, os.path

    train_categories = []
    train_samples = []
    for i in os.listdir(training_path):
        train_categories.append(i)
        train_samples.append(len(os.listdir(training_path + i)))

    test_categories = []
    test_samples = []
    for i in os.listdir(test_path):
        test_categories.append(i)
        test_samples.append(len(os.listdir(test_path + i)))

    print("Count of Fruits in Training set:", sum(train_samples))
    print("Count of Fruits in Set :", sum(test_samples))

    transform = transforms.Compose(
        [
            # transforms.RandomRotation(30),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.3),
            # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    batch_size = 256

    trainset = torchvision.datasets.ImageFolder(root=training_path, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    testset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    classes = trainset.classes

    with open("./classes.dat", "w") as file:
        for i in classes:
            file.write(f"{i}\n")

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout1 = nn.Dropout(0.25)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.dropout2 = nn.Dropout(0.3)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(14 * 14 * 256, 64)
            self.fc2 = nn.Linear(64, 128)
            self.bn5 = nn.BatchNorm1d(128)
            self.dropout3 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(128, 36)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.bn3(x)
            x = self.pool3(x)
            x = self.dropout1(x)
            x = F.relu(self.conv4(x))
            x = self.bn4(x)
            x = self.pool4(x)
            x = self.dropout2(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.bn5(x)
            x = self.dropout3(x)

            x = self.fc3(x)

            return x

    # image of NN model
    # dummy_input = torch.randn(batch_size, 3, 224, 224)
    # model = Net()
    # output = model(dummy_input)
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.format = 'png'
    # dot.render('net')

    # print("Visualization saved as 'net.png'.")

    net = Net()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    accuracies = []
    f1_scores = []
    epoch_num = 200
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics, for batch_size = 256 it takes 13 steps.
            running_loss += loss.item()
            # if i == 12:
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
            #     running_loss = 0.0
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        # only for statistics
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        predictions = []
        labels_all = []
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                predictions.extend(predicted.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(testloader)
        val_losses.append(avg_val_loss)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        f1 = f1_score(labels_all, predictions, average="weighted")
        f1_scores.append(f1)
        print(
            f"Epoch {epoch+1}/{epoch_num} Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}"
        )

    print("Finished Training")

    def plot_stats(train_losses, val_losses, accuracies, f1_scores):
        epochs = range(1, len(train_losses) + 1)

        # Losses
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/loss_plot.png")
        plt.close()

        # Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, accuracies, label="Accuracy", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/accuracy_plot.png")
        plt.close()

        # 1 Score
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, f1_scores, label="F1 Score", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("F1 Score")
        plt.title("F1 Score Over Epochs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/f1_score_plot.png")
        plt.close()

    plot_stats(train_losses, val_losses, accuracies, f1_scores)

    PATH = "./trained_net.pth"
    torch.save(net.state_dict(), PATH)

    from torch.utils.data import DataLoader

    # Create a new iterator each time:
    def get_random_batch(dataloader):
        """Gets a random batch of images and labels from the dataloader."""
        dataiter = iter(dataloader)
        try:
            return next(dataiter)
        except StopIteration:
            print("Warning: DataLoader is exhausted. Resetting...")
            dataiter = iter(dataloader)
            return next(dataiter)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # cm = confusion_matrix(labels_all, predictions)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    # disp.plot()
    # plt.show()
