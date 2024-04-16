from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
import glob
from PIL import Image, ImageEnhance
import numpy as np
import random

def enhanceBC(image, brightness, contrast):
    enhancer = ImageEnhance.Brightness(image)
    enhanced_im = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(enhanced_im)
    enhanced_im = enhancer.enhance(contrast)
    return enhanced_im


class RandomRotate90:
    def __call__(self, img):
        if random.random() > 0.5:  # 随机决定是否进行旋转
            return img.rotate(90)  # 旋转90度
        return img

def erase_and_norm(x):
    x = transforms.RandomErasing(p=1, scale=(0.0025, 0.01), ratio=(0.4, 2.5), value='random')(x)
    x = transforms.RandomErasing(p=1, scale=(0.0016, 0.0064), ratio=(0.5, 2.0), value='random')(x)
    return transforms.Normalize((0.5,), (0.5,))(x)

def normalize(x):
    return transforms.Normalize((0.5,), (0.5,))(x)


def unnormalize(x):
    return (x + 1.0) / 2.0


'''
        transforms.Lambda(lambda x: enhanceBC(x, brightness=3, contrast=2)),
        transforms.Grayscale(num_output_channels=1),
'''

data_transforms = {
    'train':transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        RandomRotate90(),
        transforms.ToTensor(),
    ]),
    'test':transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

class MyDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = np.array(img)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_DataLoader(data_path,
                   train_batch_size,
                   test_batch_size,
                   num_workers,
                   pin_memory
                   ):
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    train_fnames = glob.glob(os.path.join(train_dir, '*'))
    test_fnames = glob.glob(os.path.join(test_dir, '*'))
    train_dataset = MyDataset(train_fnames, data_transforms['train'])
    test_dataset = MyDataset(test_fnames, data_transforms['test'])
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    return train_dataloader, test_dataloader
