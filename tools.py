import os
from matplotlib import pyplot as plt
import random
import numpy as np
import jittor
import pickle

def save_features(features, features_path):
    """
    临时将提取的训练图像特征保存到文件中。

    参数:
    features (dict): 存储图像特征的字典。
    features_path (str): 保存特征的文件路径。 Dataset/features.pkl
    """
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"特征已保存到 {features_path} ")


def load_features(features_path):
    """
    从文件中加载特征

    参数:
    features_path (str): 包含特征的文件路径。

    返回:
    dict: 加载的特征字典。
    """
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    print(f"特征已从 {features_path} 加载")
    return features


