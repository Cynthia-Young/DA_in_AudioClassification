import pandas as pd
import pydub
from tqdm import tqdm
import os

def map_group_to_class_id(group):
    if group == 'us':
        return 0
    elif group == 'indian':
        return 1
    elif group == 'england':
        return 2
    else:
        return None
    
def create_new_file_name(file_path):
    return file_path.replace("cv-valid-train", "./cv-valid-train/wav").replace(".mp3", ".wav")



df = pd.read_csv('cv-valid-train.csv')
df_us = df[df['accent']=='us']
df_ind = df[df['accent']=='indian']
df_uk = df[df['accent']=='england']
df = pd.concat([df_us, df_ind, df_uk])
counts = df['text'].value_counts()

# 打印出每个唯一值的出现次数，按降序排列
print(counts[:10])


df_a = df[df['text']=='besides that there was a heap of bicycles']
df_s = df[df['text']=='i have the diet of a kid who found twenty dollars']
df_d = df[df['text']=='in spite of this i still believed that there were men in mars']
df_f = df[df['text']=='he did find it soon after dawn and not far from the sand pits']
df_g = df[df['text']=='most of them were staring quietly at the big table']
df_h = df[df['text']=='the night was warm and i was thirsty']
df_j = df[df['text']=='it had a diameter of about thirty yards']
df_k = df[df['text']=='some storms are worth the wreckage']
df = pd.concat([df_a, df_s, df_d, df_f, df_g, df_h, df_j, df_k])
# counts = df['accent'].value_counts()
# print(counts[:10])
df.drop(['text', 'up_votes', 'down_votes', 'age', 'gender', 'duration'],
        axis=1, inplace=True)
df['class_id'] = df['accent'].map(map_group_to_class_id)
df['new_file_name'] = df['filename'].apply(create_new_file_name)

new_df = df.loc[:, ['new_file_name', 'class_id', 'accent']]
new_df.columns = ['file_path', 'class_id', 'class_name']
new_df = new_df.reset_index(drop=True)

class_counts = new_df['class_id'].value_counts()
print(new_df['class_id'].value_counts())

max_count = class_counts.max()

# 复制样本数量少的类别数据，直到其数量与最多类别的数量相同
for class_id, count in class_counts.items():
    if count < max_count:
        # 计算需要复制的数量
        num_copies = max_count - count
        # 找出该类别的样本数据
        class_data = new_df[new_df['class_id'] == class_id]
        # 随机选择需要复制的样本数据，并添加到 DataFrame 中
        copied_data = class_data.sample(n=num_copies, replace=True)
        new_df = pd.concat([new_df, copied_data])
new_df = new_df.reset_index(drop=True)
print(new_df['class_id'].value_counts())

new_df.to_csv('short_df_mozilla_500.csv', index=False)