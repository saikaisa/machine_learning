import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activate(self, x):
        return np.where(x >= 0, 1, 0)

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for epoch in range(self.epochs):
            for i in range(num_samples):
                x_i = X[i, :]
                y_i = y[i]
                prediction = self.activate(np.dot(x_i, self.weights) + self.bias)
                self.weights += self.learning_rate * (y_i - prediction) * x_i
                self.bias += self.learning_rate * (y_i - prediction)

    def predict(self, X):
        return self.activate(np.dot(X, self.weights) + self.bias)


def plot_decision_boundary(X, y, perceptron):
    # 绘制数据点
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Not admitted')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='x', label='Admitted')

    # 添加坐标轴标签
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, linewidths=1, colors='green')
    plt.legend()
    plt.show()


def main():
    # 读取数据
    X = np.loadtxt('../data/ex4x.dat')
    y = np.loadtxt('../data/ex4y.dat')
    perceptron = Perceptron(learning_rate=0.0005, epochs=100000)
    perceptron.train(X, y)
    print("权重:", perceptron.weights)
    print("偏置项:", perceptron.bias)
    plot_decision_boundary(X, y, perceptron)


if __name__ == "__main__":
    main()
