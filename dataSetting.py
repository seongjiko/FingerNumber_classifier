import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader
from torchvision import models
from collections import OrderedDict
import torch.nn as nn

file_path = os.getcwd() + '/dataSet/*/*.png' # 데이터의 경로 저장
file_list = glob(file_path)

data_dict = {'image_name': [], 'class': [], 'target': [], 'file_path': []}
target_dict = {'yi_1': 1, 'er_2': 2, 'san_3': 3, 'si_4': 4, 'wu_5': 5, 'liu_6': 6, 'qi_7':7, 'ba_8': 8, 'jiu_9': 9,
               'shi_10': 10}

for path in file_list:
    data_dict['file_path'].append(path)  # file_path 항목에 파일 경로 저장

    path_list = path.split(os.path.sep)  # os(여기선 mac os)별 파일 경로 구분 문자로 split
    print(path_list)

    data_dict['image_name'].append(path_list[-1])
    data_dict['class'].append(path_list[-2])
    data_dict['target'].append(target_dict[path_list[-2]])

train_df = pd.DataFrame(data_dict)
# print('\n<data frame>\n', train_df)

train_df.to_csv(os.getcwd()+"/train.csv", mode='w')



def get_df():
    df = pd.read_csv(os.getcwd()+"/train.csv")

    # 데이터셋 train, val, test로 나누기
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=2356)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=2356)

    return df_train, df_val, df_test

class Classification_Dataset(Dataset):
    def __init__(self, csv, mode, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self,index):
        row = self.csv.iloc[index]
        image = Image.open(row.file_path).convert('RGB')
        target = torch.tensor(self.csv.iloc[index].target).long()

        if self.transform:
            image = self.transform(image)

        return image, target

dataset_train = Classification_Dataset(df_train, 'train', transform=transforms.ToTensor())

from torchvision import transforms


def get_transforms(image_size):
    global c_mean
    global c_std
    transforms_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(c_mean, c_std)])

    transforms_val = transforms.Compose([
        transforms.Resize(image_size + 30),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(c_mean, c_std)])

    return transforms_train, transforms_val
