import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from model.dcgan import DCGAN


def predict(args):

    model = DCGAN()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    model.generator.generate_image(args.out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--weights", type = str, default = 'weights/last_weights.pt')
    parser.add_argument("--out", type = str, default = 'test/image.jpg')
    args = parser.parse_args()

    predict(args)