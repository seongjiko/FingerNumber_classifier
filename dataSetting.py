import os
from glob import glob
import pandas as pd

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
print('\n<data frame>\n', train_df)

train_df.to_csv(os.getcwd()+"/train.csv", mode='w')