# 读取数据集
from typing import Any, Dict
import os
from PIL import Image
from jittor.dataset import Dataset
import jittor as jt
from jittor import nn


class TrainCustomDataset(Dataset):
    def __init__(self, features, labels, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.features = features
        self.labels = labels


    def __getitem__(self, idx):
        img, label = self.features[idx], self.labels[idx]
        if self.transform:
            img, label = self.transform(img), label
        return img, label

    def __len__(self):
        return len(self.features)






if __name__ == "__main__":
    jt.flags.use_cuda = 1

    import jclip as clip

    model, preprocess = clip.load("ViT-B-32.pkl")
    train_dir = 'Dataset/train.txt'
    class_dir = 'Dataset/classes.txt'



