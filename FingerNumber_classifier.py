# 데이터 셋 구성하기, 경로를 파악한 후
# 클래스 이름(name), 클래스(class), 그리고 학습을 위한 클래스를 숫자로 나타낸 타겟(target)을 csv파일에 저장
import os
from glob import glob # 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉터리의 리스트 반환
import pandas as pd

file_path = os.getcwd() + '/dataSet/*/*.png' # 데이터의 경로 저장
file_list = glob(file_path)

data_dict = {'image_name':[], 'class':[], 'target':[], 'file_path':[]}
# 학습에 사용하기 위한 넘버링(?)
target_dict = {'yi': 0, 'er':1, 'san':2, 'si':3, 'wu':4, 'liu':5, 'qi':6, 'ba':7, 'jiu':8, 'shi':9}

for path in file_list:
    data_dict['file_path'].append(path) # file_path 항목에 파일 경로 저장

    path_list = path.split(os.path.sep) # os별 파일 경로 구분 문자로 split

    data_dict['image_name'].append(path_list[-1]) # 이미지 이름 저장
    data_dict['class'].append(path_list[-2]) # 어떤 클래스인지 저장
    data_dict['target'].append(target_dict[path_list[-2]]) # 그 클래스의 번호 저장

train_df = pd.DataFrame(data_dict)
train_df.to_csv(os.getcwd()+"/train.csv", mode='w')


