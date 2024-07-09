## 数据集

**speech accent archive：**源码中speech accent archive音频存入在recordings文件夹，code\preprocess\speakers_all.csv是该数据集的详细信息。数据可在https://www.kaggle.com/datasets/rtatman/speech-accent-archive下载。

**Mozilla common voice：**源码中mozilla common voice音频存入在cv-valid-train文件夹，code\preprocess\cv-valid-train.csv记录了该数据集的详细信息。数据可在https://commonvoice.mozilla.org/en/datasets下载。

## 文件结构

**code\preprocess**存放了部分数据预处理操作

**code**目录下存放了预训练和适应阶段的代码

运行**code\demo** 可以用可视化界面查看模型效果，代码内可选取想要测试的模型

**test_samples** 放了两个数据集的个别数据 用以测试

**model**是训练好的模型，存为.pt格式

