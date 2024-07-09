import pandas as pd

pd.options.mode.chained_assignment = None

def clean_df(file):
    df = pd.read_csv(file)
    df_clean = df.drop(df[df['file_missing?']==True].index)
    df_clean.drop(['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11'], axis=1, inplace=True)
    mask = df_clean['country'].isin(['jamaica', 'usa', 'uk', 'india', 'canada',
    'philippines', 'singapore', 'malaysia', 'new zealand', 'south africa',
    'zimbabwe', 'namibia', 'pakistan', 'sri lanka', 'australia'])
    short_df = df_clean[mask]
    short_df.drop('file_missing?', axis=1, inplace= True)
    short_df['group'] = short_df.loc[:, 'country']
    short_df['group'].replace('pakistan', 'india', inplace=True)
    short_df['group'].replace('sri lanka', 'india', inplace=True)
    short_df['group'].replace('zimbabwe', 'south africa', inplace=True)
    short_df['group'].replace('namibia', 'south africa', inplace=True)
    short_df['group'].replace('jamaica', 'bermuda', inplace=True)
    short_df['group'].replace('trinidad', 'bermuda', inplace=True)
    short_df.drop(1771, inplace=True)
    group_mask = short_df['group'].isin(['usa', 'india', 'uk'])
    short_df = short_df[group_mask]
    return short_df

def create_new_file_name(wav):
    return './recordings/wav/{}.wav'.format(wav)

def map_group_to_class_id(group):
    if group == 'usa':
        return 0
    elif group == 'india':
        return 1
    elif group == 'uk':
        return 2
    else:
        return None
    
if __name__ == '__main__':
    df = clean_df('speakers_all.csv')
    print(df['group'].value_counts())
    df['class_id'] = df['group'].map(map_group_to_class_id)
    df['new_file_name'] = df['filename'].apply(create_new_file_name)
    new_df = df.loc[:, ['new_file_name', 'class_id', 'group']]
    new_df.columns = ['file_path', 'class_id', 'class_name']
    new_df = new_df.reset_index(drop=True)

    class_counts = new_df['class_id'].value_counts()
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
    new_df.to_csv('short_df.csv', index=False)