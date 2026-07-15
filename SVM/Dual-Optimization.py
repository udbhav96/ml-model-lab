import pandas as pd
import numpy as np


class LinearSVM:

    def __init__(self, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None


    def load_csv(self, filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except:
            print("File not loading")
            return None


    def dataTrans(self):
        df = self.load_csv("/kaggle/input/datasets/organizations/uciml/iris/Iris.csv")
        X = df[["SepalLengthCm",
                "SepalWidthCm",
                "PetalLengthCm",
                "PetalWidthCm"]].values
        y = df["Species"].values
        y[y == "Iris-setosa"] = 0
        y[y == "Iris-versicolor"] = 1
        y[y == "Iris-virginica"] = 2
        y = y.astype(int)

        return X, y


    def train_test_split(self, X, y, test_size=0.2):
        m = len(X)
        indices = np.random.permutation(m)
        test_size = int(m * test_size)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        return X_train, X_test, y_train, y_test


    def build_gram_matrix(self, X):
        return X @ X.T


    def fit(self, X, y):
        m = X.shape[0]
        K = self.build_gram_matrix(X)
        Q = (y[:, None] * y[None, :]) * K
        alpha = np.zeros(m)
        for _ in range(self.epochs):
            grad = 1 - Q @ alpha
            alpha += self.learning_rate * grad
            alpha = np.maximum(alpha, 0)
            alpha = alpha - y * (np.dot(alpha, y) / np.dot(y, y))
        self.w = ((alpha * y)[:, None] * X).sum(axis=0)
        self.b = np.mean(y - X @ self.w)


    def decision_function(self, X):
        return X @ self.w + self.b


    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)





svm = LinearSVM()

# load data
X, y = svm.dataTrans()

# split data
X_train, X_test, y_train, y_test = svm.train_test_split(X, y)
classes = [0, 1, 2]
models = []

for c in classes:
    binary_y = np.where(y_train == c, 1, -1)
    model = LinearSVM()
    model.fit(X_train, binary_y)
    models.append(model)




scores = []
for model in models:
    scores.append(model.decision_function(X_test))
scores = np.array(scores)

predictions = np.argmax(scores, axis=0)
accuracy = svm.accuracy(y_test, predictions)

print("Accuracy:", accuracy)