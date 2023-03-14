import torch

from common import ModelClass, center_crop, standard_processing, transform
from constants import MODEL_PATH, classes


def infer(image_name: str):
    model = ModelClass()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    image = standard_processing(image_name)
    image = center_crop(image, 10, 10)
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    output = model(image_tensor)
    index = output.data.cpu().numpy().argmax()
    print(classes[index])
    return index


if __name__ == '__main__':
    image_names = 'image.jpg'
    infer(image_names)
