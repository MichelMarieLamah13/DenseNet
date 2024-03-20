import argparse

import torch
from torch import nn

from DenseNetPretrain import CustomDenseNet
import yaml


def read_config(args):
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
        for key, item in config.items():
            setattr(args, key, item['value'])

    return args


def test():
    parser = argparse.ArgumentParser(description="DenseNET")
    parser.add_argument('--config',
                        type=str,
                        default="config.yml",
                        help='Configuration file')

    args = parser.parse_args()
    args = read_config(args)
    model = CustomDenseNet(name=args.name)
    input_batch = CustomDenseNet.get_batch(url=args.url)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)

    # print(output[0], flush=True)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities, flush=True)

    CustomDenseNet.read_categories(k=10, probabilities=probabilities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseNET")
    parser.add_argument('--config',
                        type=str,
                        default="config.yml",
                        help='Configuration file')

    args = parser.parse_args()
    args = read_config(args)
    model = CustomDenseNet(name=args.name)
    input_batch = CustomDenseNet.get_batch(url=args.url)

    conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    result = conv0(input_batch)

    print(result, flush=True)
    print(result.shape, flush=True)
