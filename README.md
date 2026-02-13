🧠 Hyperparameter Tuning using GridSearchCV
📌 Project Overview

This project demonstrates model optimization using GridSearchCV on the Breast Cancer Wisconsin Dataset.

The objective is to improve classification performance by tuning hyperparameters of machine learning models using cross-validation.

This project compares:
✅ Default Model Performance
✅ Tuned Model Performance
✅ Cross-Validation Results
✅ Performance Improvement Analysis

🎯 Objective

To understand and implement:
Hyperparameter tuning using GridSearchCV
Model optimization techniques
Cross-validation strategies
Performance comparison between default and tuned models
Model export and reproducibility

📊 Dataset Information

Dataset Used: Breast Cancer Wisconsin Dataset (Built-in Scikit-learn Dataset)
Total Samples: 569
Features: 30 numerical features
Target Classes:
0 → Malignant
1 → Benign
The dataset is used for binary classification.

🛠 Tools & Technologies

Python
Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn
Joblib
Google Colab

🔬 Methodology
1️⃣ Data Loading
Loaded dataset using sklearn.
Converted to Pandas DataFrame for inspection.

2️⃣ Data Splitting
Train-Test Split (80:20)
Stratified sampling to preserve class distribution.

3️⃣ Default Model Training
Trained Random Forest with default parameters.
Evaluated baseline accuracy.

4️⃣ Pipeline Creation (Unique Approach)
A pipeline was created including:
StandardScaler
RandomForestClassifier
This ensures:
Clean workflow
No data leakage
Reproducibility

5️⃣ Hyperparameter Tuning
Used GridSearchCV with:
5-Fold Cross Validation
Multiple hyperparameter combinations
Parallel processing (n_jobs = -1)

6️⃣ Model Evaluation
Compared:
Default Model Accuracy
Tuned Model Accuracy
Confusion Matrix
Classification Report

7️⃣ Model Saving
Automatically saved:
Best Parameters
Classification Report
Model Comparison Table
Trained Model (.pkl)

📂 Project Structure
Hyperparameter-Tuning-GridSearch/
│
├── notebook.ipynb
├── model_comparison.csv
├── best_parameters.txt
├── classification_report.txt
├── tuned_model.pkl
└── README.md

#Outputs
<img width="1467" height="340" alt="Image" src="https://github.com/user-attachments/assets/591aa37e-8764-43d7-84f0-8649fe74d8b7" />
<img width="396" height="231" alt="Image" src="https://github.com/user-attachments/assets/29c61693-df12-4fff-b7be-e565700539cb" />
<img width="755" height="549" alt="Image" src="https://github.com/user-attachments/assets/ef51067b-c793-48cc-a275-e27f5bcfe9a2" />
