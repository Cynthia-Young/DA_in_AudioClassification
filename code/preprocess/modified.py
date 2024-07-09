import os
import pandas as pd

# 指定目录路径
directory = 'cv-valid-train/wav'

# 读取 CSV 文件
df = pd.read_csv('cv-valid-train.csv')
df['new_column'] = 0
# 遍历目录中的文件
for filename in os.listdir(directory):
    if filename.endswith(".wav"):
        # 构建文件路径
        file_path = "cv-valid-train/" + filename.replace(".wav", ".mp3")
        # 在 CSV 文件中查找对应的行
        row = df[df['filename'] == file_path]
        if not row.empty:
            # 将新列设置为1
            # print(filename)
            df.loc[row.index, 'new_column'] = 1

# 保存修改后的 CSV 文件
df.to_csv('cv-valid-train_modified.csv', index=False)
