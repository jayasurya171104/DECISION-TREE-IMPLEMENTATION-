# DECISION-TREE-IMPLEMENTATION-

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JAYASURYA M

*INTERN ID*: CT04DF2950

*DOMAIN*: MACHINE LEARNING 

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

# DESCRIOTON OF MY TASK 
 :

🧠 CodTech Internship - Task 1: Decision Tree Implementation
📌 Objective:
Build and visualize a Decision Tree model using Scikit-learn to classify or predict outcomes based on the Titanic dataset.

📁 Dataset:
Source: Titanic Dataset (Titanic-Dataset.csv)

Target variable: Survived (0 = No, 1 = Yes)

Features used: Passenger class, Age, Fare, Gender, Embarkment port, etc.

⚙️ Technologies Used:
Python

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

🔍 Steps Performed:
Data Loading & Cleaning:

Loaded the dataset using pandas.

Removed rows with missing values (dropna()).

Dropped irrelevant columns: Name, Ticket, Cabin.

Feature Engineering:

Used one-hot encoding to convert categorical variables (Sex, Embarked) into numeric format using pd.get_dummies().

Model Preparation:

Defined feature matrix X and target variable y.

Split data into training and testing sets using train_test_split() (80/20 split).

Model Training:

Trained a DecisionTreeClassifier on the training set.

Evaluation:

Evaluated the model on the test set using classification_report and accuracy_score.

Visualization:

Visualized the decision tree using plot_tree() to understand how the model makes decisions.

📊 Output:
Model Accuracy: (Output shown in the console)

Classification Report: (Precision, Recall, F1-score shown in the console)

Decision Tree Plot: Graphical representation of the trained decision tree.

📒 Deliverable:
A well-documented Python Notebook or .py script including:

Preprocessing steps

Model training & testing

Tree visualization

Summary of results



*OUTPUT*

![Image](https://github.com/user-attachments/assets/0b163a11-0a7e-4b1b-b1fb-a96eb7279285)
