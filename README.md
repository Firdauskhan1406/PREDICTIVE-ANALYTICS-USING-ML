# PREDICTIVE-ANALYTICS-USING-ML
"COMAPNY" : CODTECH IT SOLUTIONS

"NAME" : FIRDAUS KHAN

"INTERN ID" : CT08NJP

"DOMAIN": DATA ANALYTICS

"DURATION: 4 WEEKS

"MENTOR" : NEELA SANTOSH

##OUTPUT OF THE CODES 
![Image](https://github.com/user-attachments/assets/3e3d8326-607e-42bb-b251-8cca2314ab71)

![Image](https://github.com/user-attachments/assets/2bc4d70f-16e2-4d92-aff1-a0b271a56b0a)

![Image](https://github.com/user-attachments/assets/d675729e-4c81-409d-909c-c0f6f8fbda5a)

![Image](https://github.com/user-attachments/assets/d0b79768-7afd-4c1b-a786-06fd6e38495a)

![Image](https://github.com/user-attachments/assets/9c2738b3-b436-48ef-93eb-89789ebeb797)




Objective
The Titanic dataset survival prediction project aims to:


Predict whether a passenger survived the Titanic disaster using machine learning models.

Use passenger features such as age, sex, class, fare, and embarked port to determine survival likelihood.

Steps in the Project

1. Dataset Overview
Dataset: Titanic dataset (train.csv and optionally test.csv from Kaggle).
Features:
Pclass: Passenger class (1st, 2nd, or 3rd).
Sex: Gender (male or female).
Age: Passenger age.
Fare: Ticket price.
SibSp and Parch: Number of siblings/spouses and parents/children aboard.
Embarked: Port of embarkation (C, Q, S).
Target Variable: Survived (0 = No, 1 = Yes).


3. Data Preprocessing
Handling Missing Values:
Replace missing Age values with the median age.
Replace missing Embarked with the mode.
Handle missing Fare values if present.
Feature Encoding:
Convert categorical features (Sex, Embarked) into numerical values using techniques like one-hot encoding or label encoding.
Scaling Features:
Normalize features like Fare to bring them into a comparable range using StandardScaler or MinMaxScaler.


5. Exploratory Data Analysis (EDA)
Analyze survival rates by different features (Pclass, Sex, etc.).
Visualize relationships using bar plots, histograms, and box plots.
Check feature correlations with survival.


7. Model Building
Data Splitting:
Split the dataset into training and testing sets using train_test_split.
Model Selection:
Train multiple machine learning models, such as:
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Gradient Boosting (e.g., XGBoost or LightGBM)
Hyperparameter Tuning:
Optimize model performance using Grid Search or Random Search with cross-validation.


9. Model Evaluation
Evaluate model performance using metrics such as:
Accuracy: Percentage of correct predictions.
Confusion Matrix: Counts of true positives, true negatives, false positives, and false negatives.
Precision, Recall, F1-Score: Assess prediction quality.
ROC-AUC: Measure model's ability to distinguish between classes.


11. Feature Importance
Analyze which features contribute most to survival predictions (e.g., Sex, Pclass).
Evaluate model performance using classification metrics.



Conclusion
Insights:

Passengers in 1st class were more likely to survive compared to those in 3rd class.
Female passengers had higher survival rates than males.
Younger passengers had a higher chance of survival.
Higher ticket prices (Fare) were associated with increased survival likelihood.
Model Performance:

Logistic Regression provides a simple and interpretable model.
Random Forest and Gradient Boosting models often achieve higher accuracy due to their ability to capture complex patterns.




STEPS IN THE PROJECT IN DETAIL 
Steps in the Project
1. Dataset Understanding
Features: Includes Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
Target Variable: Survived (1 for survival, 0 for non-survival).
Key relationships, like Sex and Pclass with survival rates, are explored.


3. Data Preprocessing
Handling Missing Values: Missing Age values filled with the median; Embarked filled with the mode.
Encoding Categorical Features: Sex and Embarked converted to numerical values using Label Encoding or One-Hot Encoding.
Feature Scaling: Fare and Age normalized using StandardScaler for uniform feature contribution.


5. Exploratory Data Analysis (EDA)
Visualizations such as bar plots and histograms reveal:
Women and children had higher survival rates.
Passengers in 1st class were more likely to survive compared to those in 3rd class.
High fares often indicated a greater chance of survival.


7. Model Building
Data Splitting: Dataset split into training and test sets (e.g., 80:20 split).
Models Used:
Logistic Regression for baseline prediction.
Random Forest and Gradient Boosting for better performance with complex patterns.
Support Vector Machines (SVM) or k-Nearest Neighbors (kNN) as alternative options.
Pipeline Creation: Combines preprocessing and model training steps for efficiency and consistency.


9. Model Evaluation
Metrics used:
Accuracy: Measures the overall correctness of predictions.
Confusion Matrix: Displays true positives, true negatives, false positives, and false negatives.
Classification Report: Shows precision, recall, and F1-score.
ROC-AUC Score: Evaluates the model’s ability to distinguish between classes.


11. Feature Importance
Features like Sex, Pclass, and Fare significantly impact predictions.
Positive coefficients indicate an increase in survival probability; negative ones indicate a decrease.
Conclusions
Women, children, and 1st-class passengers had higher survival rates.
Socio-economic factors (e.g., Fare) influenced survival likelihood.
Random Forest and Gradient Boosting models outperformed Logistic Regression in accuracy and generalization.




Outcome
This project demonstrates how machine learning can identify patterns in data and provide actionable insights for decision-making. The Titanic dataset serves as an excellent case study for binary classification tasks, feature engineering, and model evaluation techniques.
