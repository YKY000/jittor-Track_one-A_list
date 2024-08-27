import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
import numpy as np
from extract_Feature import textold_feature, KM_select_four_img, extract_representative_features_by_mean
from tools import load_features
from CustomDataset import TrainCustomDataset

import pdb
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

best_dir = '/root/autodl-tmp/Dataset/output/best_mean_cutmix_color.pkl'
# best_dir = 'Dataset/last.pkl'
checkpoint = jt.load(best_dir)
model.load_state_dict(checkpoint["model"])
print("epoch",checkpoint["epoch"])

split = 'TestSet' + args.split
imgs_dir = '/root/autodl-tmp/Dataset/' + split  # Dataset/TestSetA
test_imgs = os.listdir(imgs_dir)

num_classes,text_features = textold_feature(class_dir) # 提取所有种类文本
aug_text_features = jt.array(text_features)

# testing data processing
print('Testing data processing:')
test_features = []
extract_test_features = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)
        test_features.append(image)

        # 模型提取特征
        image_features = model.encode_image(image)
        # 对图像特征进行归一化处理
        image_features /= image_features.norm(dim=-1, keepdim=True)
        extract_test_features.append(image_features)
print("测试数据特征提取完成")


print('Representing data processing:')
features = load_features(features_path)
new_train_imgs, new_train_labels, remaining_imgs, remaining_labels = KM_select_four_img(features)
train_imgs_dir = '/root/autodl-tmp/Dataset/'
all_four_img_features = []
represent_features = []
with jt.no_grad():
    for img in tqdm(new_train_imgs):
        img = os.path.join(train_imgs_dir, img)
        image = Image.open(img)
        preprocessed_image = preprocess(image).unsqueeze(0)  # [1,3,224,224,]
        all_four_img_features.append(preprocessed_image)

all_four_img_features = jt.array(jt.concat(all_four_img_features, dim=0))
all_four_img_labels = jt.array(jt.concat(new_train_labels, dim=0))
all_dataset = TrainCustomDataset(all_four_img_features, all_four_img_labels)
all_loader = DataLoader(all_dataset, batch_size=16, shuffle=True, num_workers=0)
represent_features = extract_representative_features_by_mean(model, all_loader)
represent_features = jt.array(jt.cat(represent_features))
print("代表数据特征提取完成")

extract_test_features = jt.array(jt.cat(extract_test_features))   # 提取特征的测试图片
test_features = jt.array(jt.cat(test_features))     # 未提取特征的测试图片
with jt.no_grad():
    logits_per_image, logits_per_text = model(test_features,aug_text_features)
    # 计算余弦相似度
    cosine_similarity = extract_test_features @ represent_features.t()

    cosine_similarity = cosine_similarity.softmax(dim=-1)
    probs = logits_per_image.softmax(dim=-1)
    preds = (cosine_similarity + probs).softmax(dim=-1).tolist()


# testing
with open('result.txt', 'w') as save_file:
    i = 0
    for prediction in preds:
        prediction = np.asarray(prediction) # prediction是个列表，里面是374个种类概率
        top5_idx = prediction.argsort()[-1:-6:-1]
        save_file.write(test_imgs[i] + ' ' +
                        ' '.join(str(idx) for idx in top5_idx) + '\n')
        i += 1
print("任务完成")