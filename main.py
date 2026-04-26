import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("processed_dataset.csv")

df['Transported'] = df['Transported'].astype(int)

target_reg = 'FoodCourt' #признак для регресии

target_clf = 'Transported' #признак джля квалификации

X = df.drop(columns=[target_reg, target_clf])

y_reg = df[target_reg]
y_clf = df[target_clf]

X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.3, random_state=42
)

reg_model = DecisionTreeRegressor(max_depth=10, random_state=42) #регрессия
reg_model.fit(X_train, y_reg_train)

y_reg_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("=== РЕГРЕССИЯ ===")
print("MSE:", mse)
print("R2:", r2)

clf_model = DecisionTreeClassifier(max_depth=5, random_state=42) #классификация
clf_model.fit(X_train, y_clf_train)

y_proba = clf_model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_clf_test, y_proba[:, 1]) #roc-кривая
roc_auc = auc(fpr, tpr)

print("\n=== КЛАССИФИКАЦИЯ ===")
print("ROC-AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, marker='o', label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', label="Random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC-кривая")
plt.legend()
plt.grid()
plt.show()