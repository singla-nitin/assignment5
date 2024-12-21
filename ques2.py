import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


file_path = '/content/drive/MyDrive/Colab Notebooks/DATASETS/weather.csv'
weather_data = pd.read_csv(file_path)

categorical_features = ['Outlook', 'Temp', 'Humidity']
binary_features = ['Windy']
target = 'Play'

encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(weather_data[categorical_features])

X = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
X['Windy'] = weather_data[binary_features].values
y = weather_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

decision_tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)

plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
