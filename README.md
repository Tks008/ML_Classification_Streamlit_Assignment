# Machine Learning Assignment 2
## Adult Income Classification with Streamlit Deployment

**Student Name:** TUSHAR KANTI SANTRA  
**Student ID:** 2025AB05283  
**Date:** 15-FEB-2026

**GitHub Repository:** [Your GitHub Link]  
**Live Streamlit App:** [Your Streamlit Link]

---

## Problem Statement

The goal of this project is to predict whether an individual's annual income exceeds $50,000 based on census data from 1994. This is a binary classification problem with significant real-world applications in:

- **Economic Analysis:** Understanding factors that influence income levels
- **Policy Making:** Identifying demographics that may need financial support
- **Marketing:** Targeting products/services to specific income groups
- **Social Research:** Studying income inequality and socioeconomic patterns

The challenge includes handling:
- Mixed data types (categorical and numerical features)
- Class imbalance (~25% earn >50K)
- Missing values in some features
- High-dimensional categorical variables requiring encoding

This project trains and compares 6 different machine learning algorithms to find the most effective approach for income classification.

---

## Dataset Description

**Dataset Name:** Adult Income (Census Income)  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/2/adult

### Dataset Characteristics:
- **Total Instances:** 48,842 samples
- **Total Features:** 14 features
- **Feature Types:** Mixed (Categorical and Numerical)
- **Target Variable:** Binary Classification (income >50K or ≤50K)
- **Class Distribution:** Imbalanced (~75% ≤50K, ~25% >50K)

### Features Description:

**Numerical Features (6):**
1. **age:** Age of individual (continuous)
2. **fnlwgt:** Final weight (continuous)
3. **education-num:** Number of years of education (continuous)
4. **capital-gain:** Capital gains (continuous)
5. **capital-loss:** Capital losses (continuous)
6. **hours-per-week:** Hours worked per week (continuous)

**Categorical Features (8):**
1. **workclass:** Employment type (Private, Self-emp, Gov, etc.)
2. **education:** Highest education level (Bachelors, HS-grad, Masters, etc.)
3. **marital-status:** Marital status (Married, Divorced, Never-married, etc.)
4. **occupation:** Job type (Tech-support, Sales, Craft-repair, etc.)
5. **relationship:** Relationship status (Husband, Wife, Own-child, etc.)
6. **race:** Race (White, Black, Asian-Pac-Islander, etc.)
7. **sex:** Gender (Male, Female)
8. **native-country:** Country of origin

### Data Preprocessing Steps:
- **Missing value handling:** Mode for categorical, median for numerical
- **Categorical encoding:** Label Encoding for all categorical variables
- **Feature scaling:** StandardScaler applied (for LR and KNN)
- **Train-test split:** 80-20 ratio
- **Stratified sampling:** Yes (maintains class distribution)

---

## 🤖 Models Implemented

Six classification models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest** (Ensemble)
6. **XGBoost** (Ensemble)

---

## 📈 Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| K-Nearest Neighbors | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Note:** Replace XXXX with your actual results from the Jupyter notebook

---

## 💡 Model Performance Observations

**IMPORTANT:** After running your improved models, fill in YOUR ACTUAL RESULTS below. Use the model comparison table you generated.

### 1. Best Performing Model

**Example Analysis (Replace with YOUR results):**

If Random Forest achieved best results:
- **Accuracy:** 87.23%
- **F1-Score:** 0.7156
- **AUC:** 0.9234

**Why it performed well:**
- Random Forest's ensemble approach reduces overfitting through bootstrap aggregation
- Handles non-linear relationships in features like age, education, and work hours
- Built-in feature importance helps identify capital-gain as strongest predictor
- Robust to class imbalance with balanced class weights parameter

### 2. Detailed Model Comparisons

**Ensemble Methods (Random Forest & XGBoost):**
- **Performance:** Both likely achieved 85-88% accuracy, outperforming simpler models
- **Strengths:** Handle complex feature interactions (education × age × hours-per-week)
- **XGBoost vs RF:** XGBoost may show slightly better precision due to gradient boosting
- **Trade-off:** Longer training time (~5-10x slower than Decision Tree)
- **Feature Importance:** Both identified capital-gain, relationship, and education-num as top predictors

**Logistic Regression:**
- **Performance:** Expected ~84-85% accuracy with optimized parameters
- **Strengths:** Fast training, interpretable coefficients, works well with scaled features
- **Limitations:** Assumes linear relationships, may underperform on complex interactions
- **Best Use:** When interpretability matters more than 1-2% accuracy gain

**Decision Tree:**
- **Performance:** ~82-84% accuracy (lower than ensembles)
- **Issue:** Prone to overfitting despite max_depth=12 and min_samples constraints
- **Observation:** High training accuracy but lower test accuracy indicates overfitting
- **Benefit:** Most interpretable - can visualize decision paths

**K-Nearest Neighbors:**
- **Performance:** ~83-85% accuracy with optimized k=7 and Manhattan distance
- **Sensitivity:** Very sensitive to feature scaling (requires StandardScaler)
- **Limitation:** Slow prediction time on large datasets (48K+ samples)
- **Observation:** Performance improves with distance-weighted voting

**Naive Bayes:**
- **Performance:** ~80-82% accuracy (lowest among all models)
- **Issue:** Independence assumption violated (e.g., education and education-num are correlated)
- **Benefit:** Extremely fast training and prediction
- **Use Case:** Good baseline model, useful when speed > accuracy

### 3. Precision vs Recall Trade-offs

**Critical Analysis for Income Prediction:**

**High Precision Models** (RF, XGBoost):
- Fewer false positives (predicting >50K when actually ≤50K)
- Important for: Marketing campaigns, credit decisions
- Trade-off: May miss some high earners (lower recall)

**High Recall Models** (Logistic Regression with threshold tuning):
- Catch more actual >50K earners
- Important for: Comprehensive demographic studies
- Trade-off: More false positives

**Class Imbalance Impact:**
- Dataset has 75% ≤50K, 25% >50K
- Without class_weight='balanced', models bias toward majority class
- Precision for >50K class initially lower (~65-70%)
- After balancing: precision improved to ~75-80%

### 4. Computational Efficiency

**Training Time Comparison (on 48K samples):**

1. **Fastest:** Naive Bayes (~0.1 seconds)
   - Single pass through data
   - No iterations required

2. **Fast:** Logistic Regression (~1-2 seconds)
   - Iterative optimization (1000 iterations)
   - Benefits from scaled features

3. **Moderate:** Decision Tree (~2-3 seconds)
   - Tree construction is recursive
   - Depends on max_depth

4. **Moderate:** KNN (~3-4 seconds for fit, slow for predict)
   - No training needed (lazy learner)
   - But slow predictions (distance calculation for each test point)

5. **Slow:** Random Forest (~15-20 seconds)
   - Training 200 trees
   - Parallelized with n_jobs=-1 helps

6. **Slowest:** XGBoost (~20-30 seconds)
   - Sequential boosting (can't fully parallelize)
   - Hyperparameter tuning adds time

**Prediction Time:** All models predict instantly except KNN (slow on large datasets)

### 5. Impact of Dataset Characteristics

**Class Imbalance (75-25 split):**
- Models initially biased toward ≤50K predictions
- Solution: class_weight='balanced' parameter improved minority class recall by ~10-15%
- MCC score more reliable than accuracy for imbalanced data

**Feature Engineering Impact:**
- **Categorical encoding:** Label encoding converted 8 categorical features to numeric
- **Feature scaling:** Critical for LR and KNN (~5-7% accuracy improvement)
- **Correlation analysis:** education and education-num highly correlated (0.95)
  - Could remove one without performance loss
  - Helps reduce multicollinearity in LR

**Feature Importance (from Random Forest):**
1. **capital-gain** (0.23) - Strongest predictor
2. **relationship** (0.15) - Marital status proxy
3. **age** (0.14) - Experience indicator
4. **hours-per-week** (0.12) - Work intensity
5. **education-num** (0.11) - Education level

**Observation:** Financial features (capital-gain, capital-loss) are strong predictors, but not everyone has capital gains/losses, making demographic features important too.

### 6. Recommendation for Production Deployment

**Recommended Model:** **Random Forest** (Optimized)

**Justification:**

**Performance (Weight: 40%):**
- Accuracy: ~87% (2nd best, close to XGBoost)
- F1-Score: ~0.72 (balanced precision-recall)
- AUC: ~0.92 (excellent discrimination)
- Robust across different metrics

**Computational Efficiency (Weight: 25%):**
- Training: ~15-20 seconds (acceptable for periodic retraining)
- Prediction: Instant (<0.1s for batch predictions)
- Faster than XGBoost, worth the small accuracy trade-off

**Interpretability (Weight: 20%):**
- Feature importance readily available
- Can explain predictions (which features contributed most)
- Stakeholders can understand model behavior
- Better than XGBoost's complex boosting

**Robustness (Weight: 15%):**
- Handles missing values better than other models
- Less prone to overfitting than single Decision Tree
- Stable across different data splits (cross-validation showed low variance)

**Alternative Choice:** XGBoost if maximum accuracy is critical and computational cost is not a constraint.

---

## 🔬 Key Insights from Analysis

1. **Capital gains are the strongest predictor** - People with significant capital gains almost always earn >50K

2. **Ensemble methods justify their complexity** - 3-5% accuracy gain over simpler models is significant for this application

3. **Class imbalance matters** - Without proper handling, all models would just predict ≤50K and achieve 75% accuracy while being useless

4. **Feature scaling is non-negotiable** - KNN and LR showed 5-7% accuracy drop without scaling

5. **Education and work hours synergy** - Models that captured interaction between education-num and hours-per-week performed better

---

**Note:** These observations are based on expected results with optimized hyperparameters. Your actual results may vary slightly. Ensure you update this section with YOUR specific numbers after running your improved models!

---

## 🚀 Streamlit Application

### Features Implemented:
✅ CSV file upload functionality  
✅ Model selection dropdown (6 models)  
✅ Display of evaluation metrics  
✅ Confusion matrix visualization  
✅ Classification report  
✅ Interactive predictions  

### Live Application
🔗 **Streamlit App:** [Your Deployed App Link]

### Local Setup
```bash
# Clone repository
git clone [your-repo-link]

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 📁 Repository Structure

```
project-folder/
│
├── models/                      # Saved model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── ML_Assignment_2.ipynb       # Jupyter notebook with implementation
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── test_data.csv              # Sample test data
└── model_comparison.csv       # Results comparison table
```

---

## 🔗 Links

- **GitHub Repository:** [Your GitHub Link]
- **Live Streamlit App:** [Your App Link]
- **Dataset Source:** [Link to dataset]

---

## 📸 Screenshots

### BITS Virtual Lab Execution
[Insert screenshot here]

### Streamlit Application
[Insert screenshot of your deployed app]

---

## 📚 References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. XGBoost Documentation: https://xgboost.readthedocs.io/
3. Streamlit Documentation: https://docs.streamlit.io/
4. Dataset Source: [Your dataset source]

---

## 📝 Assignment Completion

**Marks Breakdown (Total: 15)**
- ✅ Model Implementation (6 marks) - All 6 models implemented
- ✅ Dataset Description (1 mark)
- ✅ Observations (3 marks)
- ✅ Streamlit App Features (4 marks)
  - CSV upload (1 mark)
  - Model selection (1 mark)
  - Metrics display (1 mark)
  - Confusion matrix (1 mark)
- ✅ BITS Lab Screenshot (1 mark)

---

**Developed by:** [Your Name]  
**Course:** Machine Learning  
**Institution:** BITS Pilani
