# -------- what will I do ? ----------
# ** I will download and create DeepLabV3 model for your personal use case.
# ------------------------------------

# --------------------------------------
# Neural Network imports
# --------------------------------------
from torchvision import models
from models.segmentation.deeplabv3 import DeepLabHead



def createDeepLabv3(outputchannels=1):

    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)

    # Added a Tanh activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)

    # Set the model in training mode
    # model.train()

    return model


model = createDeepLabv3()

# print("Model Created")
