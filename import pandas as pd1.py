
import pandas as pd1
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load the data and remove the extra quotes around column names/values
# quoting=3 tells pandas to ignore the quote marks in the file
df = pd1.read_csv('bank-additional-full.csv', sep=';', quoting=3)

# 2. Clean up column names (remove any leftover " marks)
df.columns = df.columns.str.replace('"', '')
# Clean up the data values (remove " from the text)
df = df.replace('"', '', regex=True)

# 3. Preprocessing: Create dummy variables
df_final = pd1.get_dummies(df, drop_first=True)

# 4. Define Features (X) and Target (y)
# Now that quotes are gone, 'y_yes' will be found correctly
X = df_final.drop('y_yes', axis=1) 
y = df_final['y_yes']

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize and Train the Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = clf.predict(X_test)
print(f"Model Prediction Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 8. Identify Top 5 Economic Drivers
importances = pd1.Series(clf.feature_importances_, index=X.columns)
print("\nTop 5 Drivers of Consumer Decision:")
print(importances.sort_values(ascending=False).head(5))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, precision=2)
plt.savefig('bank_decision_tree.png')
plt.show()
plt.close('all')