<<<<<<< HEAD
# Algorithm Challenge 2026 解决方案说明

本项目用于完成 `iftechio/algorithm-challenge-2026` 题目：预测用户在首次播放后一个月内是否会重复播放某个节目，并输出对应的概率。

根据题目要求，本仓库重点提供以下三部分内容：

- 特征工程代码：[feature_engineering.py](./feature_engineering.py)
- 模型训练代码：[train_model.py](./train_model.py)
- 使用说明文档：[README.md](./README.md)

当前最终采用的模型是 `DeepFM`，因为它在本地验证集上取得了最好的验证效果。

## 一、 `split_dataset.py` 和 `feature_engineering.py` 的区别

这两个文件职责不同：

- `split_dataset.py`
  - 作用是把 `train.csv` 切分成训练集和验证集
  - 它只负责“数据划分”
  - 不负责构造模型输入特征

- `feature_engineering.py`
  - 作用是构造模型真正使用的特征
  - 包括用户侧特征、节目侧特征、统计特征、交叉统计特征、hashed sparse 特征
  - 它才是这个项目里的“特征工程代码”


## 二、项目结构

- `feature_engineering.py`
  - 读取用户特征表和节目特征表
  - 构造 dense 统计特征
  - 构造 sparse 类别特征
  - 提供模型训练所需的通用特征函数

- `train_model.py`
  - 最终提交模型训练入口
  - 当前采用的模型是 `DeepFM`
  - 支持训练、验证以及测试集预测

- `split_dataset.py`
  - 将 `train.csv` 切分为本地训练集和验证集

- `predict_deepfm.py`
  - 使用已经训练好的 `DeepFM` checkpoint 生成最终提交文件

## 三、数据准备

请将原始数据放在如下目录：

```text
data/
  train.csv
  test.csv
  user_feature.csv
  episode_feature.csv
```

## 四、本地验证集构造

为了在本地评估模型效果，需要先从 `train.csv` 中划分训练集和验证集。

推荐命令：

```powershell
python .\split_dataset.py --write-subsets
```

执行后会生成：

```text
data/splits/stratified_train.csv
data/splits/stratified_val.csv
```

本项目最终主要使用 `stratified` 切分结果进行离线验证。

## 五、特征工程说明

特征工程代码位于 [feature_engineering.py](./feature_engineering.py)。

主要包含以下几类特征：

### 1. 用户侧特征

- `address`
- `age`
- `sex`
- `rg_source`
- `membership_days`

### 2. 节目侧特征

- `duration_minutes`
- `primary_category`
- `language`
- `category_count`

### 3. 统计类特征

- `uid_rate`
- `episode_rate`
- `tab_rate`
- `scene_rate`
- `entrance_rate`

这些特征本质上是基于训练数据统计得到的目标相关特征。

### 4. 交叉统计特征

- `uid x tab_name`
- `uid x scene_name`
- `uid x entrance_type`
- `episode_id x tab_name`
- `primary_category x entrance_type`
- `language x tab_name`

这些特征用于增强用户、内容和场景之间的组合关系建模能力。

### 5. 稀疏类别特征

项目还对类别字段进行了哈希映射，供神经网络模型的 embedding 层使用。

## 六、模型训练

最终训练脚本是 [train_model.py](./train_model.py)。

当前最终模型为 `DeepFM`。

推荐训练命令如下：

```powershell
conda run -n pytorch_gpu python .\train_model.py --epochs 1 --batch-size 2048 --predict-test
```

该命令会完成：

- 读取 `data/splits/stratified_train.csv`
- 读取 `data/splits/stratified_val.csv`
- 构造特征
- 训练 `DeepFM`
- 在验证集上评估模型
- 如指定 `--predict-test`，则生成测试集预测结果

## 七、生成提交文件

如果已经有训练好的 DeepFM 模型，可以直接执行：

```powershell
conda run -n pytorch_gpu python .\predict_deepfm.py --model-path .\artifacts\deepfm_full_bs2048\model.pt --output-csv .\result.csv --batch-size 2048
```

生成的提交文件格式如下：

```csv
id,label
0,0.45
1,0.73
2,1.00
```

>>>>>>> 52cf1be (feature：提交主要代码)
