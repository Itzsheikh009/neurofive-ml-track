# Titanic Survival Prediction — Logistic Regression Classifier

## 📌 Project Overview
This machine learning project predicts passenger survival on the Titanic based on demographic and ticket features using a **Logistic Regression** model built with `scikit-learn`.

---

## 🛠️ Workflow & Methodology
1. **Data Cleaning & Preprocessing:**
   * Imputed missing values for numerical features (`Age`, `Fare`) using median values.
   * Filled missing categorical values (`Embarked`) using the mode.
2. **Feature Encoding:**
   * Applied One-Hot Encoding (`pd.get_dummies`) to binary and multi-class categorical variables (`Sex`, `Embarked`) with `drop_first=True` to prevent multicollinearity.
3. **Data Splitting:**
   * Split dataset into **80% training** and **20% testing** sets using stratified sampling to preserve target class balance.
4. **Model Training & Evaluation:**
   * Trained a `LogisticRegression` classifier.
   * Evaluated model accuracy and analyzed error distribution via a confusion matrix.

---

## 📊 Results & Key Metrics
* **Final Test Accuracy:** `~80%`
* **Confusion Matrix Insights:**
  * High precision in identifying non-survivors.
  * Majority of prediction errors stem from false negatives (under-predicting survival due to dataset class imbalance).

---

## 🚀 How to Run Locally

```bash
# Clone repository
git clone [https://github.com/your-username/titanic-ml-classifier.git](https://github.com/your-username/titanic-ml-classifier.git)
cd titanic-ml-classifier

# Install dependencies
pip install pandas numpy scikit-learn

# Run script
python train_model.py
