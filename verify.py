import torch
from torchvision import datasets

from common import (BATCH_SIZE, DIRECTORY, MODEL_PATH, NUM_WORKERS, ModelClass,
                    transform)


def verify(model=None):
    if not model:
        model = ModelClass()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

    test_data = datasets.ImageFolder(f'{DIRECTORY}/test', transform=transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # test model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')


if __name__ == '__main__':
    verify()
