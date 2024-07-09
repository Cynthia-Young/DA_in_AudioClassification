import pandas as pd
import pydub
from tqdm import tqdm

pd.options.mode.chained_assignment = None

def clean_df(file):
    df = pd.read_csv(file)
    # df_us = df[df['accent']=='us'].sample(13470)
    # df_ind = df[df['accent']=='indian']
    # df_uk = df[df['accent']=='england'].sample(13470)
    # df = pd.concat([df_us, df_ind, df_uk])
    # df = df[df['new_column']==1]
    df.drop(['text', 'up_votes', 'down_votes', 'age', 'gender', 'duration'],
        axis=1, inplace=True)
    # df.to_csv('short_df_mozilla.csv', index=False)
    return df

def mp3towav(df, col):
    for filename in tqdm(df[col]):
        pydub.AudioSegment.from_mp3("cv-valid-train/{}.mp3".format(filename)).export("cv-valid-train/wav/{}.wav".format(filename), format="wav")

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
    

# 354, 293, 61
if __name__ == '__main__':
    df = clean_df('cv-valid-train.csv')
    print(df['accent'].value_counts())
    df['class_id'] = df['accent'].map(map_group_to_class_id)
    df['new_file_name'] = df['filename'].apply(create_new_file_name)
    mp3towav(df, 'new_file_name')

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

    new_df.to_csv('short_df_mozilla.csv', index=False)