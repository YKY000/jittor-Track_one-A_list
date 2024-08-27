import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import numpy as np
from extract_Feature import textold_feature, extract_img_feature, KM_select_four_img, extract_representative_features_by_mean
from CustomDataset import TrainCustomDataset
from Augmentation import Blur, adjust_contrast, cutmix, cutout, apply_data_augmentation, color_jitter
from tools import save_features, load_features

from jittor.dataset import DataLoader
import shutil

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

model, preprocess = clip.load("ViT-B-32.pkl")
class_dir = '/root/autodl-tmp/Dataset/classes.txt'
train_dir = '/root/autodl-tmp/Dataset/train.txt'
features_path = '/root/autodl-tmp/Dataset/img_feature/features.pkl'

if not os.path.exists(features_path):
    features = extract_img_feature(model, train_dir, preprocess)  # 用clip提取图像特征,所有图像归一化后
    save_features(features, features_path)
else:
    features = load_features(features_path)


new_train_imgs, new_train_labels, remaining_imgs, remaining_labels = KM_select_four_img(features)  # 1496

num_classes, text_features = textold_feature(class_dir)  # 提取所有种类文本

imgs_dir = '/root/autodl-tmp/Dataset/'


all_four_img_features = []
train_features = []
cutmix_augmented_features = []
colorjitter_augmented_features = []
print('Training data processing:')
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0)  # [1,3,224,224,]
        train_features.append(preprocessed_image)
        all_four_img_features.append(preprocessed_image)

        cutmix_image_features = apply_data_augmentation(image, cutmix, preprocess)
        cutmix_augmented_features.append(cutmix_image_features)
        colorjitter_image_features = apply_data_augmentation(image, color_jitter, preprocess)
        colorjitter_augmented_features.append(colorjitter_image_features)
print("特征训练完成")


train_features.extend(cutmix_augmented_features)
train_features.extend(colorjitter_augmented_features)  # 合并特征列表
train_features = jt.array(jt.concat(train_features, dim=0))  # [2992,3,224,224,]
# train_labels = jt.array(jt.concat(new_train_labels, dim=0))
train_labels = jt.array(jt.concat(new_train_labels + new_train_labels + new_train_labels, dim=0))  # [2992,]
aug_text_features = jt.array(text_features)

all_four_img_features = jt.array(jt.concat(all_four_img_features, dim=0))
all_four_img_labels = jt.array(jt.concat(new_train_labels, dim=0))
all_dataset = TrainCustomDataset(all_four_img_features, all_four_img_labels)


print('Validating data processing:')
val_features = []
with jt.no_grad():
    for img in tqdm(remaining_imgs):
        img = os.path.join(imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0)  # [1,3,224,224,]
        val_features.append(preprocessed_image)
val_features = jt.array(jt.concat(val_features, dim=0))
val_labels = jt.array(jt.concat(remaining_labels, dim=0))

# 构造数据集
train_dataset = TrainCustomDataset(train_features, train_labels)
val_dataset = TrainCustomDataset(val_features, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=0)
all_loader = DataLoader(all_dataset, batch_size=16, shuffle=False, num_workers=0)

print("数据集构造完成")


criterion = jt.nn.CrossEntropyLoss()
optimizer = jt.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.0001)  # weight_decay：1e-4 2.22 lr：1e-2
lr_scheduler = jt.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30], gamma=0.1)
# lr_scheduler = jt.lr_scheduler.CosineAnnealingLR(optimizer, 20 * len(train_loader))


represent_features = extract_representative_features_by_mean(model, all_loader)
represent_features = jt.array(jt.cat(represent_features))
print("代表数据特征提取完成")

print('Training data processing:')
num_epochs = 40
print_every = 1
best_acc = 0.0
contrastive_loss_weight = 1.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):  # inputs:[16,3,224,224,] labels:[16,1,]
        # 前向传播
        logits_per_image, logits_per_text = model(inputs,aug_text_features)  # logits_per_image:[16,374,] labels:[16,1,] logits_per_text:[374,16,]

        inputs_features = model.encode_image(inputs)
        inputs_features /= inputs_features.norm(dim=-1, keepdim=True)
        cosine_similarity = inputs_features @ represent_features.t()
        logits_per_image = (cosine_similarity + logits_per_image)
        loss = criterion(logits_per_image, labels)

        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.clip_grad_norm(0.1, 2)
        optimizer.step()

        running_loss += loss.data[0]
    lr_scheduler.step()

    # 打印训练损失
    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # 验证循环
    model.eval()
    with jt.no_grad():
        total_correct = 0
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            logits_per_image, logits_per_text = model(inputs, aug_text_features)  # [8,374,]

            inputs_features = model.encode_image(inputs)
            inputs_features /= inputs_features.norm(dim=-1, keepdim=True)
            cosine_similarity = inputs_features @ represent_features.t()
            logits_per_image = (cosine_similarity + logits_per_image)

            max_indices, max_probs = jt.argmax(logits_per_image, 1)  # [8,]
            labels = jt.flatten(labels)
            correct_count = (max_indices == labels).sum().item()
            total_correct += correct_count

        acc_valid = total_correct / len(val_loader.dataset)
        print(f"Validation Accuracy: {acc_valid * 100:.3f}%")

    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
    }
    jt.save(checkpoint, f"/root/autodl-tmp/Dataset/output/last_mean_cutmix_color.pkl")

    if best_acc < acc_valid:
        # 保存验证集上表现最好的模型
        best_acc, best_epoch = acc_valid, epoch
        shutil.copy(f"/root/autodl-tmp/Dataset/output/last_mean_cutmix_color.pkl",
                    f"/root/autodl-tmp/Dataset/output/best_mean_cutmix_color.pkl")

print(f"best_acc: {best_acc}")
print(f"best_epoch: {best_epoch}")
