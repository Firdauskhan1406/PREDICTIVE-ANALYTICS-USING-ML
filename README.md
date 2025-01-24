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

Dataset Overview
The Titanic dataset contains the following columns:

Survived: The target variable indicating survival (1 for survived, 0 for not survived).

Pclass: The passenger class (1st, 2nd, or 3rd).

Sex: Gender of the passenger.

Age: Age of the passenger in years.

SibSp: Number of siblings or spouses aboard the Titanic.

Parch: Number of parents or children aboard the Titanic.

Fare: Ticket price paid by the passenger.

Embarked: The port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Some columns have missing values (Age, Cabin, and occasionally Embarked), which need to be handled during preprocessing.



The Titanic survival prediction problem is a well-known challenge in machine learning. The task involves analyzing data from the Titanic disaster to determine which passengers were more likely to survive. 

This dataset is widely used in machine learning competitions and education because it provides a balanced mix of numerical and categorical data, making it ideal for learning preprocessing, feature engineering, and model evaluation techniques.

The dataset includes details such as passenger age, gender, class, fare, and embarkation port. Using this information, machine learning models can be trained to predict survival outcomes. 


This process involves multiple steps, including data cleaning, exploration, feature transformation, model training, and evaluation.


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


Step 1: Importing Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
These libraries are essential for data manipulation, preprocessing, visualization, and machine learning.

Step 2: Data Loading and Exploration
Load the dataset using Pandas:

python
Copy
Edit
data = pd.read_csv('titanic.csv')
print(data.head())
print(data.info())
print(data.describe())
Check for missing values:

python
Copy
Edit
print(data.isnull().sum())
Step 3: Handling Missing Data
Age: Replace missing values with the median of the column.
Embarked: Replace missing values with the mode.
Cabin: Drop this column since it has too many missing values.
python
Copy
Edit
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(['Cabin'], axis=1, inplace=True)
Step 4: Feature Engineering
Convert categorical columns into numerical format:

python
Copy
Edit
data['Sex'] = LabelEncoder().fit_transform(data['Sex'])  # 0 for female, 1 for male
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
Normalize continuous features like Age and Fare:

python
Copy
Edit
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])
Step 5: Splitting Data
Split the data into features and target variables:

python
Copy
Edit
X = data.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Step 6: Model Building
Train a Random Forest classifier:

python
Copy
Edit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
Step 7: Predictions
Make predictions on the test set:

python
Copy
Edit
y_pred = model.predict(X_test)
Step 8: Model Evaluation
Evaluate the model’s performance:

python
Copy
Edit
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
Visualize the confusion matrix:

python
Copy
Edit
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
Step 9: Feature Importance
Analyze the importance of each feature:

python
Copy
Edit
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print(feature_importance_df)

# Visualization
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
Insights and Observations
Key Features:

Sex (female passengers were more likely to survive).
Pclass (1st-class passengers had higher survival rates).
Fare (higher ticket prices indicated greater chances of survival).
Model Performance:

Accuracy: ~80-85% depending on the test set and random state.
The model effectively differentiates between survivors and non-survivors using the provided features.
Patterns in Data:

Women and children had a much higher survival rate than


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
