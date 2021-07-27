from tkinter import *

from webcamgui import App
from imutils.video import FPS

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, utils
import torchvision.transforms as transforms

from resnet_classes import class_names

# load pretrained resnet model
model = models.resnet18(pretrained=True)
model.eval()

# some transforms
transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])

fps = FPS().start()
root = Tk()
App(root, fps, model=model, transforms=transforms, class_names = class_names, window_title="Imagenet Inference")
root.mainloop()
fps.stop()
print("[INFO] elapsed time: {: .2f}".format(fps.elapsed()))
print("[INFO] approx FPS: {: .2f}".format(fps.fps()))