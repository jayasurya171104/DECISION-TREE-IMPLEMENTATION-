

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# %%
data = pd.read_csv("/content/Titanic-Dataset.csv")
data.dropna(inplace=True)
# Drop non-numeric columns that are not needed for the model
data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
data = pd.get_dummies(data, columns=["Sex", "Embarked"], drop_first=True)
X = data.drop("Survived", axis=1)
y = data["Survived"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=["Not Survived", "Survived"])
plt.show()
