import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KernelSVM:

    def __init__(self, learning_rate=0.001, epochs=1000, gamma=1):
        self.lr = learning_rate
        self.epochs = epochs
        self.gamma = gamma
    # RBF Kernel
    def kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)

    # Gram Matrix
    def build_gram_matrix(self, X):
        m = X.shape[0]
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = self.kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        m = X.shape[0]
        K = self.build_gram_matrix(X)
        Q = (y[:, None] * y[None, :]) * K
        alpha = np.zeros(m)
        for _ in range(self.epochs):
            grad = 1 - Q @ alpha
            alpha += self.lr * grad
            alpha = np.maximum(alpha, 0)
            alpha = alpha - y * (np.dot(alpha, y) / np.dot(y, y))
        self.alpha = alpha
        # bias
        decision = (alpha * y) @ K
        self.b = np.mean(y - decision)

    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for i in range(len(self.alpha)):
                s += (
                    self.alpha[i]
                    * self.y_train[i]
                    * self.kernel(self.X_train[i], x)
                )
            preds.append(np.sign(s + self.b))
        return np.array(preds)




X, y = make_moons(n_samples=200, noise=0.1)

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


svm = KernelSVM(learning_rate=0.001, epochs=2000, gamma=5)

svm.fit(X_train, y_train)



preds = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))



def plot_decision_boundary(model, X, y):

    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(X[:,0], X[:,1], c=y)

    plt.show()


plot_decision_boundary(svm, X, y)