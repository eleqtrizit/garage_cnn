import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from common import ModelClass, transform
from constants import BATCH_SIZE, DIRECTORY, MODEL_PATH, NUM_WORKERS
from verify import verify

# https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
EPOCHS = 10
# use gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    model = ModelClass()

    train_data = datasets.ImageFolder(f'{DIRECTORY}/train', transform=transform)

    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        print(f'Epoch {epoch + 1} of {EPOCHS}...')
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), MODEL_PATH)
    verify()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
