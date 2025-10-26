import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import glob, os, joblib

X, y = [], []
for file in glob.glob("data/*.npy"):
    X.append(np.mean(np.load(file), axis=0))
    y.append(os.path.basename(file).split("_")[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "sign_model.pkl")
