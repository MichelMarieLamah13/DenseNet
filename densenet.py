import os

import torch
import urllib.request

from PIL import Image
from torchvision import transforms

import torch.nn as nn


class CustomDenseNet(nn.Module):
    """
    Names: densenet169, densenet201, densenet161
    """

    def __init__(self, name):
        super(CustomDenseNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model=name, pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def get_batch(url, folder='images'):
        """
        To download image
        :param url: the url of the image
        :param folder: the folder where to store the image
        :return:
        """
        filename = CustomDenseNet.download_data(
            folder=folder,
            url=url
        )
        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        return input_batch

    @staticmethod
    def download_data(url, folder='images'):
        os.makedirs(folder, exist_ok=True)
        name = os.path.basename(url)
        filename = f"{folder}/{name}"
        if not os.path.exists(filename):
            try:
                urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print("Une erreur s'est produite lors du téléchargement du fichier:", e)

        return filename

    @staticmethod
    def read_categories(k, probabilities):
        """
        Get top categories
        :param k: number of categories with highest probabilities
        :param probabilities: the probabilities
        :return:
        """
        filename = CustomDenseNet.download_data(
            url='https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
        )
        with open(filename, "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        topk_prob, topk_catid = torch.topk(probabilities, k)
        for i in range(topk_prob.size(0)):
            print(categories[topk_catid[i]], topk_prob[i].item())
