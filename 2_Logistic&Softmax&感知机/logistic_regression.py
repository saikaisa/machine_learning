import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    # 绘制数据点
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Not admitted')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='x', label='Admitted')

    # 设置x,y轴范围
    plt.xlim(10, 70)
    plt.ylim(40, 100)

    # 添加坐标轴标签
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')


def plot_line_logistic(X, y, model, color, name):
    # 获取数据的最小值和最大值
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    # 计算决策边界的直线
    x_values = np.array([x_min, x_max])
    y_values = -(model.theta[0] + model.theta[1] * x_values) / model.theta[2]

    # 绘制决策边界
    plt.plot(x_values, y_values, color, label=name)


class LogisticRegression:
    def __init__(self, learning_rate=0.001, epoch=200000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.theta = None

    def sigmoid(self, x):
        # Sigmoid函数
        return 1 / (1 + np.exp(-x))

    def gradient_descent(self, X, y):
        # 梯度下降法
        num_samples, num_features = X.shape
        # theta 的第一个参数：偏置项。第二个参数：Exam1的系数。第三个参数：Exam2的系数
        self.theta = np.zeros(num_features)
        # 梯度下降进行参数更新
        for _ in range(self.epoch):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, h - y) / num_samples
            self.theta = self.theta - self.learning_rate * gradient

    def stochastic_gradient_descent(self, X, y):
        # 随机梯度下降法
        num_samples, num_features = X.shape
        self.theta = np.zeros(num_features)
        for _ in range(self.epoch):
            for i in range(num_samples):
                h = self.sigmoid(np.dot(X[i], self.theta))
                gradient = np.dot(X[i].T, (h - y[i])) / num_samples
                self.theta -= self.learning_rate * gradient

    def newton_method(self, X, y):
        # 牛顿法
        num_samples, num_features = X.shape
        self.theta = np.zeros(num_features)
        for _ in range(self.epoch):
            h = self.sigmoid(np.dot(X, self.theta))
            gradient = np.dot(X.T, (h - y)) / num_samples
            Hessian = np.dot(X.T, np.diag(h * (1 - h))).dot(X) / num_samples
            self.theta -= np.dot(np.linalg.inv(Hessian), gradient)

    def fit(self, X, y, method):
        # 添加一列常数项 1，作为偏置项，添加在 X 的第 0 列
        X = np.insert(X, 0, 1, axis=1)
        if method == 'gradient_descent':
            self.gradient_descent(X, y)
        elif method == 'stochastic_gradient_descent':
            self.stochastic_gradient_descent(X, y)
        elif method == 'newton_method':
            self.newton_method(X, y)
        else:
            raise ValueError('Invalid method!')


def main():
    # 读取数据
    X = np.loadtxt('../data/ex4x.dat')
    y = np.loadtxt('../data/ex4y.dat')

    # 初始化模型
    model = LogisticRegression()

    # 绘制数据点
    plot_data(X, y)

    # 使用梯度下降法训练模型
    model.fit(X, y, method='gradient_descent')
    print(f"梯度下降法得到的参数向量：{model.theta}")

    # 绘制梯度下降法决策边界
    plot_line_logistic(X, y, model, color="gray", name="GD")

    # 使用随机梯度下降法训练模型
    model.fit(X, y, method='stochastic_gradient_descent')
    print(f"随机梯度下降法得到的参数向量：{model.theta}")

    # 绘制随机梯度下降法决策边界
    plot_line_logistic(X, y, model, color="pink", name="SGD")

    # 使用牛顿法训练模型
    model.fit(X, y, method='newton_method')
    print(f"牛顿法得到的参数向量：{model.theta}")

    # 绘制牛顿法决策边界
    plot_line_logistic(X, y, model, color="green", name="Newton")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
