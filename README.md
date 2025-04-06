# 📊Telecom Churn Prediction with R 

A machine learning project to predict customer churn in the telecom industry using **Random Forest** and **Decision Tree** classifiers in **R**. The project also explores **entropy calculations** for deeper insights into information gain.

---

## 🔍 Objective

Customer churn significantly affects telecom companies' revenue. By identifying customers likely to leave, companies can take proactive measures to retain them.

This project includes:
- 🧠 Predictive modeling using Random Forest and Decision Tree  
- 📊 Entropy calculation (original & modified)  
- ⚖️ Class balancing using oversampling (SMOTE technique)

---

## 🧰 Tech Stack & Libraries

- 💻 **R Language**
- 📦 Libraries: `caret`, `randomForest`, `rpart`, `rpart.plot`, `ROSE`, `dplyr`, `ggplot2`, `caTools`

---

## 📊 Dataset Overview

- Source: Telco Customer Churn dataset  
- Features: Customer demographics, services used, tenure, charges, etc.  
- Target: **Churn** (Yes/No)

---

## 🔄 Workflow

### 🧹 Data Preprocessing

- Removed missing values using `na.omit`  
- Dropped unnecessary columns: `customerID`, `tenure`, `TotalCharges`  
- Grouped `tenure` into custom bins like `0-12 Month`, `12-24 Month`, etc.

### ⚖️ Handling Imbalanced Data

- Used `ROSE::ovun.sample` for oversampling minority class  
- Verified balance with proportional bar plots

### 📏 Entropy Calculation

- ✅ Original entropy formula: `-∑p * log2(p)`  - `Entropy: 0.834741`
- 🔁 Modified entropy formula: `-3.222 * ∑p * log10(p)`  - `Entropy: 0.809641`
- Compared both entropy values for the `Churn` column

### 🌳 Model Building

#### Decision Tree
- Trained with `caret::train()` and 10-fold cross-validation  
- Visualized using `rpart.plot`  
- Evaluated using `confusionMatrix()`

#### Random Forest
- *(Implemented using `randomForest` package — details in the script)*

---

## 📈 Evaluation Metrics

- Accuracy  - `0.7645`
- Confusion Matrix  - 
                                          
               Accuracy : 0.7645          
                 95% CI : (0.7491, 0.7793)
      No Information Rate : 0.5002          
      P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.5289
      Mcnemar's Test P-Value : 0.1198
            Sensitivity : 0.7506          
            Specificity : 0.7783          
         Pos Pred Value : 0.7721          
         Neg Pred Value : 0.7572          
             Prevalence : 0.5002          
         Detection Rate : 0.3754 
         Detection Prevalence : 0.4863          
         Balanced Accuracy : 0.7645
         Positive' Class : No 
                                         
- Precision / Recall (if needed)

---

## ▶️ How to Run

1. 📥 Clone the repository:
   ```bash
   git clone https://github.com/Shubham-Tambei9/Telecom-Churn-Prediction.git
   ```

2. 📂 Change Directory:
   ```bash
   cd Telecom-Churn-Prediction
   ```

