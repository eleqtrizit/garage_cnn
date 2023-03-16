import os
import sys
from pathlib import Path

import torch

from common import ModelClass, center_crop, standard_processing, transform
from constants import MODEL_PATH, classes


def infer(image_name: str):
    model_path = os.getenv('MODEL_PATH') or MODEL_PATH
    model = ModelClass()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    image = standard_processing(image_name)
    image = center_crop(image, 10, 10)
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()

    return index


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    # get the last image in the directory
    image = sorted(directory.glob('*.jpg'))[-1]
    class_index = infer(str(image))
    print(classes[class_index])
