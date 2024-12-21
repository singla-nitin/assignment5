import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

file_path = '/content/drive/MyDrive/Colab Notebooks/DATASETS/BankNote_Authentication.csv'
dataset = pd.read_csv(file_path)

X = dataset.drop(columns=['class'])
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

dt_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['Fake', 'Authentic'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
