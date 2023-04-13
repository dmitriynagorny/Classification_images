import time
import torch
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()

    losses = {'train': [], 'valid': []}
    accuracy = {'train': [], 'valid': []}
    pbar = trange(num_epochs, desc='Epoch:')
    best_acc = 0

    for epoch in pbar:
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    optimizer.zero_grad()

                if phase == 'valid':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                preds = torch.argmax(outputs, -1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_correct += int(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracy[phase].append(epoch_acc)

            pbar.set_description(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    time_elapsed = time.time() - since
    print(f'Training time is {int(time_elapsed // 60)} min {int(time_elapsed % 60)} sec')
    print(f'Best accuracy is {round(best_acc, 4)}')

    model.load_state_dict(best_model_wts)
    return model, losses, accuracy


def evaluate(model, dataloader, device, dataset_size):
    model.eval()

    running_correct = 0
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)
        _, predicted = torch.max(output, 1)

        running_correct += int(torch.sum(predicted == labels))

    return running_correct / dataset_size


def pic_losses(losses, accuracy, names: list):
    sns.set(style='whitegrid', font_scale=1.4)

    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(losses[names[0]], label=names[0])
    plt.plot(losses[names[1]], label=names[1])
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(accuracy[names[0]], label=names[0])
    plt.plot(accuracy[names[1]], label=names[1])
    plt.legend()

    plt.show()


def freeze(model, num_layers):
    for param in model.features[:-num_layers].parameters():
        param.requires_grad = False