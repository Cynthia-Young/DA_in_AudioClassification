import pandas as pd
import librosa
import librosa.display

#忽略警告
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None


# 读取 CSV 文件
csv_path = 'code/short_df_mozilla_500.csv'
df = pd.read_csv(csv_path)

# 初始化计数器
short_audio_count = 0

# 遍历 CSV 文件中的 file_path 列
for file_path in df['file_path']:
    # 使用 librosa 加载音频文件并获取时长
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 统计时长小于5秒的音频文件
    if duration < 3.5:
        short_audio_count += 1

# 打印结果
print(f"Number of audio files shorter than 5 seconds: {short_audio_count}")