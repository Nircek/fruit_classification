# Fruit Classification

A group project developed for the Image Analysis course, written in Python.
The goal was to classify different fruit types using neural networks.

![image](https://github.com/user-attachments/assets/c5c37c69-fbdc-4f56-bdc2-e6cefca31b6f)

## Usage
- Add an image using the **"+"** button.  
- Once an image is added to the dataset, the model automatically predicts its class and stores this information in the memory.  
- Users can browse through all images in the dataset using the **previous/next** buttons.
  
## Installation
Below is an example script for installing and running the program on Ubuntu 24.04.1.
```
sudo apt install python3.12-venv python3-gi python3-gi-cairo gir1.2-gtk-4.0
sudo apt install libgirepository-2.0-dev gcc libcairo2-dev pkg-config python3-dev
sudo apt install libgirepository1.0-dev python3-pip
python3 -m venv projekt-venv
source projekt-venv/bin/activate
pip install opencv-python pycairo PyGObject
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python3 app.py
```
