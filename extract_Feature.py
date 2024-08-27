import jittor as jt
import jclip as clip
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import random


def extract_representative_features_by_mean(model, dataloader):
    features = []
    labels = []

    # 提取所有样本的特征和标签
    for inputs, label in dataloader:
        features.append(model.encode_image(inputs).detach())
        labels.append(label)

    features = jt.concat(features)
    labels = jt.concat(labels)

    # 按类别分组特征
    unique_labels = np.unique(labels.numpy())
    representative_features = []

    for label in unique_labels:
        idx = (labels == label).nonzero()
        idx_first_column = idx[:, 0]
        class_features = features[idx_first_column]  # [4,512,]

        # 计算每个类别特征的平均值
        mean_features = class_features.mean(dim=0, keepdim=True)
        mean_features /= mean_features.norm(dim=-1, keepdim=True)
        representative_features.append(mean_features)

    return representative_features


def KM_select_four_img(features):
    new_train_imgs = []
    new_train_labels = []
    remaining_imgs = []
    remaining_labels = []
    for label in features:
        # 获取图像特征
        data = features[label]
        # 提取图像路径和特征
        img_paths, img_features = zip(*data)
        # 将图像特征堆叠成一个矩阵，方便进行聚类分析
        img_features = np.vstack(img_features)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(img_features)
        # 获取聚类中心点
        centers = kmeans.cluster_centers_

        # 遍历每个聚类中心，找到与其最近的图像，并将其路径和标签添加到新的训练数据集中
        num_images_per_label = 0  # 用于统计当前类别的图片数量
        selected_indices = []
        for center in centers:
            cluster_indices = np.where(kmeans.labels_ == num_images_per_label)[0]
            cluster_features = img_features[cluster_indices]
            cluster_centroid = np.mean(cluster_features, axis=0)
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_features - cluster_centroid, axis=1))]

            new_train_imgs.append(img_paths[closest_idx])
            new_train_labels.append(jt.float32([label]))
            selected_indices.append(closest_idx)

            num_images_per_label += 1  # 增加当前类别的图片数量
            if num_images_per_label >= 4:  # 控制每个类别只选择四张图片
                break
        remaining_indices = [idx for idx in range(4) if idx not in selected_indices]
        remaining_imgs.extend([img_paths[idx] for idx in remaining_indices])
        remaining_labels.extend([jt.float32([label]) for _ in range(len(remaining_indices))])
    return new_train_imgs, new_train_labels, remaining_imgs, remaining_labels


def textold_feature(class_dir):
    classes = open(class_dir).read().splitlines()
    num_classes = len(classes)
    new_classes = []
    for c in classes:  # 374
        c = c.split(' ')[0]
        if c.startswith('Animal'):
            c = c[7:]
            template = 'a photo of a {}, a type of animal.'
            template = template.format(c)
        elif c.startswith('Thu-dog'):
            c = c[8:]
            template = 'a photo of a {}, a type of dog.'
            template = template.format(c)
        elif c.startswith('Caltech-101'):
            c = c[12:]
            template = 'a photo of a ' + c
        elif c.startswith('Food-101'):
            c = c[9:]
            template = 'a photo of a {}, a type of food.'
            template = template.format(c)
        new_classes.append(template)
    text = clip.tokenize(new_classes)
    return num_classes, text


def extract_img_feature(model, train_dir, transform=None):
    imgs_dir = '/root/autodl-tmp/Dataset/'
    train_labels = open(train_dir).read().splitlines()
    train_imgs = [l.split(' ')[0] for l in train_labels]
    # train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]
    train_labels = [int(l.split(' ')[1]) for l in train_labels]  # 将标签转换为整数

    features = {}
    for i, img_path in enumerate(tqdm(train_imgs)):
        label = train_labels[i]
        img = os.path.join(imgs_dir, img_path)
        image = Image.open(img)
        preprocessed_image = transform(image).unsqueeze(0)
        image_features = model.encode_image(preprocessed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if label not in features:
            features[label] = []
        features[label].append((img_path, image_features.numpy()))  # 字典存放是图像路径(不包含Dataset)和图像特征
    return features


def extract_img_feature_four(model, train_dir, transform=None):
    imgs_dir = '/root/autodl-tmp/Dataset/'
    train_labels = open(train_dir).read().splitlines()
    train_imgs = [l.split(' ')[0] for l in train_labels]
    # train_labels = [jt.float32([int(l.split(' ')[1])]) for l in train_labels]
    train_labels = [int(l.split(' ')[1]) for l in train_labels]  # 将标签转换为整数

    features = {}
    for i, img_path in enumerate(tqdm(train_imgs)):
        label = train_labels[i]
        img = os.path.join(imgs_dir, img_path)
        image = Image.open(img)
        preprocessed_image = transform(image).unsqueeze(0)
        # image_features = model.encode_image(preprocessed_image)
        # image_features /= image_features.norm(dim=-1, keepdim=True)

        if label not in features:
            features[label] = []
        features[label].append((img_path, preprocessed_image.numpy()))  # 字典存放是图像路径(不包含Dataset)和图像特征
    return features


if __name__ == "__main__":
    from tools import load_features, save_features

    jt.flags.use_cuda = 1
    model, preprocess = clip.load("ViT-B-32.pkl")
    # class_dir = '/root/autodl-tmp/Dataset/classes.txt'
    class_dir = '/root/autodl-tmp/Dataset/classes.txt'
    train_dir = '/root/autodl-tmp/Dataset/train.txt'
    features_path = '/root/autodl-tmp/Dataset/img_feature/features.pkl'

    if not os.path.exists(features_path):
        features = extract_img_feature(model, train_dir, preprocess)  # 用clip提取图像特征
        save_features(features, features_path)
    else:
        features = load_features(features_path)
    new_train_imgs, new_train_labels, remaining_imgs, remaining_labels = KM_select_four_img(features)
    print(len(new_train_imgs), len(new_train_labels), len(remaining_imgs), len(remaining_labels))

    # rep_train_imgs, rep_train_labels = KM_select_one_img(new_train_imgs, new_train_labels)



