import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])


classes = []


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def loadClasses():
    with open("./classes.dat", "r") as file:
        loaded = [line.strip() for line in file]
    return loaded


def loadNet():
    loaded_model = Net()
    loaded_model.load_state_dict(torch.load("./trained_net.pth", weights_only=True))
    return loaded_model


def prepareImage(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize((100, 100)),
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
        print(f"Predicted class: {classes[predicted.item()]}")


if __name__ == "__main__":
    classes = loadClasses()
    net = loadNet()
    net.eval()

    image_path = "./sand.jpg"
    useNet(net, prepareImage(image_path))
