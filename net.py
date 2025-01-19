import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


classes = []


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


def loadClasses():
    with open("./classes.dat", "r") as file:
        loaded = [line.strip() for line in file if line.strip()]
    return loaded


def loadNet():
    loaded_model = Net()
    loaded_model.load_state_dict(
        torch.load(
            "./trained_net.pth", weights_only=True, map_location=torch.device("cpu")
        )
    )
    return loaded_model


def prepareImage(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


def useNet(NN, tensor_image):
    with torch.no_grad():
        output = NN(tensor_image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]


classes = loadClasses()


def predict(image_path):
    net = loadNet()
    net.eval()
    return useNet(net, prepareImage(image_path))


if __name__ == "__main__":
    print(predict("./sand.jpg"))
