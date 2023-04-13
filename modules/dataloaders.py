from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms
import os
import torch


def dataload(path, batch_size=8):

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(244),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    image_datasets['valid'] = ConcatDataset([image_datasets['valid'], image_datasets['test']])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
                   for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}

    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names