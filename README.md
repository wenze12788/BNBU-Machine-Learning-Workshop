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
pip install -r requirements.txt
```

### Core Libraries
```
pandas
numpy
scikit-learn
xgboost
```

---

## 📂 Project Structure
```
/workshop/
├── train.csv               # Training data
├── test.csv                # Test data
├── main.ipynb              # Full pipeline code
├── README.md               # Project documentation
├── submission_random_forest.csv   # RF submission
├── submission_logistic_regression.csv  # LR submission
└── other model submissions
```

---

## 🚀 How to Run / Train
### 1. Prepare Data
1. Download `train.csv` and `test.csv` from Kaggle
2. Place them in your project folder (e.g., `Desktop/workshop/`)
3. Update the path in code if needed:
   ```python
   workshop_path = r'C:\Users\shomo\Desktop\workshop'
   ```

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

---

