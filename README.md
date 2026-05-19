# 🚀 Spaceship Titanic - Kaggle Competition
**Predict which passengers are transported to an alternate dimension**

This project is a solution for the **Spaceship Titanic** Kaggle competition, where we build machine learning models to classify whether passengers were transported to an alternate dimension during a collision with a spacetime anomaly.

## 📌 Project Overview
- **Competition**: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
- **Task**: Binary classification (Transported: True / False)
- **Models Used**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- **Pipeline**: Feature Engineering → Encoding → Imputation → Scaling → Model Training → Evaluation → Submission

---

## 🛠️ Environment & Dependencies
### Requirements
- Python 3.8+
- All dependencies are listed in `requirements.txt`

### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost seaborn
```

### Core Libraries
```
pandas
numpy
scikit-learn
xgboost
seaborn
```

---

## 📂 Project Structure
```
BNBU-Machine-Learning-Workshop/
├── train.csv                 # Original training dataset
├── test.csv                  # Original test dataset
├── logistic_regression.ipynb # Logistic Regression model pipeline
├── knn.ipynb                 # KNN model pipeline
├── random_forest.ipynb       # Random Forest model pipeline
├── SVM.ipynb                 # SVM model pipeline
├── xgboost.ipynb             # XGBoost model pipeline
└── README.md                 # Project documentation
```

---

## 🚀 How to Run / Train
### 1. Prepare Data
1. Download `train.csv` and `test.csv` from Kaggle
2. Place them in your project folder (e.g., `Desktop/workshop/`)
3. Update the path in code if needed:


### 2. Run the Full Pipeline
Execute the notebook/script in order:
1. Load data
2. Feature engineering
3. Preprocessing (encoding, missing value imputation, scaling)
4. Train each model
5. Cross-validation
6. Generate submission files

### 3. Train a Model Example (Random Forest)
```python
# Train model
rf_model = RandomForestClassifier(...)
rf_model.fit(X_train, y_train)

# Evaluate
cv_scores = cross_val_score(...)
print(cv_scores.mean())

# Predict & Save submission
pred = rf_model.predict(X_test)
sub.to_csv("submission.csv")
```

---

## 🧪 Preprocessing Pipeline
1. **Feature Engineering**
   - Split PassengerId → Group / Number
   - Extract Deck / Side / CabinNum from Cabin
   - Create GroupSize, IsAlone, TotalSpent, IsVIP
2. **Categorical Encoding**: LabelEncoder
3. **Missing Values**: Median imputation
4. **Scaling**: StandardScaler (for KNN, LR, SVM)

---

## 📊 Models & Performance
All models evaluated with **5-fold cross-validation**:
- Logistic Regression
- KNN
- Random Forest
- SVM
- XGBoost

You can compare CV accuracy to select the best model for submission.

---

## 📤 Generate Submission
The code automatically generates Kaggle-ready submission files:
- `submission_logistic_regression.csv`
- `submission_random_forest.csv`
- And more for other models

Upload the CSV to Kaggle to get your score!

---

## 📝 Author
- Kaggle: Spaceship Titanic Solution
- GitHub: [Your Username]

# Spaceship Titanic 竞赛项目中文 README

# 🚀 Spaceship Titanic \- 太空泰坦尼克号 分类竞赛

本项目是 Kaggle **Spaceship Titanic** 经典二分类竞赛解决方案，通过机器学习模型预测太空乘客是否被时空异常传送，完成乘客状态分类任务。

## 📌 项目简介

- **竞赛任务**：二分类任务，预测乘客是否被传送（Transported：True / False）

- **解决方案**：完整数据清洗 \+ 特征工程 \+ 数据预处理 \+ 多模型训练对比 \+ 交叉验证 \+ 结果提交

- **使用模型**：逻辑回归、KNN近邻、随机森林、SVM支持向量机、XGBoost

---

## 💻 运行环境与依赖

### 环境要求

- Python 3\.8 及以上

- Windows / Mac / Linux 通用

### 核心依赖库

项目所有依赖：

- pandas、numpy（数据处理）

- scikit\-learn（传统机器学习模型、预处理、交叉验证）

- xgboost（梯度提升树模型）

### 一键安装依赖

```Plain Text
pip install pandas numpy scikit-learn xgboost seaborn
```

---

## 📁 项目目录结构

```Plain Text
BNBU-Machine-Learning-Workshop/  # BNBU 机器学习工作坊（项目根目录）
├── train.csv                 # 原始训练数据集（用于模型训练）
├── test.csv                  # 原始测试数据集（用于模型评估/预测）
├── logistic_regression.ipynb # 逻辑回归模型完整流程代码
├── knn.ipynb                 # K近邻模型完整流程代码
├── random_forest.ipynb       # 随机森林模型完整流程代码
├── SVM.ipynb                 # 支持向量机模型完整流程代码
├── xgboost.ipynb             # XGBoost 模型完整流程代码
└── README.md                 # 项目说明文档（介绍、使用方法、结果等）

---

## ⚙️ 数据预处理与特征工程

本项目针对竞赛数据集做了精细化特征构造，大幅提升模型效果：

1. **身份特征拆分**：拆分 PassengerId 为 Group 乘客组、Number 组内编号

2. **群体特征构造**：统计同组人数 GroupSize、是否单人出行 IsAlone

3. **船舱特征拆解**：从 Cabin 字段提取 Deck 甲板、Side 左右舷、CabinNum 舱位编号

4. **消费特征融合**：汇总各项服务消费，生成总消费特征 TotalSpent

5. **高阶特征**：根据甲板位置构造 VIP 标识 IsVIP

统一预处理流程：

- 类别特征：LabelEncoder 统一编码（训练集\+测试集拟合，避免维度偏差）

- 缺失值处理：中位数填充

- 特征标准化：StandardScaler（适配线性模型、KNN、SVM）

---

## 🤖 使用的模型介绍

项目采用 **5种经典机器学习模型** 训练对比，择优提交：

1. **逻辑回归 \(Logistic Regression\)**：基线线性模型，速度快、可解释性强，加入平衡权重处理样本不均衡

2. **KNN 近邻**：基于距离的非线性模型，对局部数据特征敏感

3. **支持向量机 SVM**：高维数据分类效果优异，泛化能力稳定

4. **随机森林 Random Forest**：集成树模型，抗过拟合、可输出特征重要性

5. **XGBoost**：梯度提升树，竞赛高分常用模型，拟合能力最强

所有模型均使用**5折交叉验证（5\-Fold CV）** 评估准确率，保证结果可靠。

---

## 🚀 训练与运行教程

### 1\. 数据准备

- 从 Kaggle 官网下载 `train\.csv`、`test\.csv`

- 将文件放入项目文件夹，修改代码中的本地路径：

```Plain Text
workshop_path = r'C:\Users\shomo\Desktop\workshop'
```

### 2\. 运行流程

按照顺序执行代码即可完成全流程：

1. 读取数据集

2. 自定义特征工程函数构造新特征

3. 特征筛选、数据集划分

4. 编码、填充缺失值、标准化预处理

5. 多模型训练 \+ 交叉验证评估

6. 测试集预测并生成提交文件

### 3\. 结果输出

运行完成后自动生成 Kaggle 标准提交 CSV 文件，可直接上传竞赛官网评测。

---

## 📊 项目亮点

- 完整可复现的机器学习流水线，从原始数据到提交结果一站式完成

- 手工高价值特征工程，贴合竞赛数据逻辑

- 五种主流模型横向对比，适合学习模型差异与调参

- 严格的训练/测试集统一预处理，避免数据泄露

- 交叉验证评估，结果更稳定可信

---

## 📄 版权与说明

本项目为个人学习练习项目，仅供机器学习入门与 Kaggle 竞赛学习参考。

