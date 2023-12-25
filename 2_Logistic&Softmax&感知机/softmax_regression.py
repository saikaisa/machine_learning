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


def plot_line_softmax(X, y, model, color, name):
    # 获取模型参数
    w = model.theta[1:]
    b = model.theta[0]

    # 计算决策边界的斜率和截距
    slope = -w[0] / w[1]
    intercept = -b / w[1]

    # 计算x,y值
    x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
    y_values = slope * x_values + intercept

    # 绘制决策边界
    plt.plot(x_values, y_values, color, label=name)


class SoftmaxRegression:
    def __init__(self, learning_rate=0.001, epoch=200000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.theta = None

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def gradient_descent(self, X, y):
        num_samples, num_features = X.shape
        num_classes = y.shape[1]
        self.theta = np.zeros((num_features, num_classes))
        for _ in range(self.epoch):
            h = self.softmax(np.dot(X, self.theta))
            gradient = np.dot(X.T, h - y) / num_samples
            self.theta = self.theta - self.learning_rate * gradient

    def stochastic_gradient_descent(self, X, y):
        num_samples, num_features = X.shape
        num_classes = y.shape[1]
        self.theta = np.zeros((num_features, num_classes))
        for _ in range(self.epoch):
            for i in range(num_samples):
                xi = X[i:i+1]
                yi = y[i:i+1]
                h = self.softmax(np.dot(xi, self.theta))
                gradient = np.dot(xi.T, (h - yi)) / num_samples
                self.theta -= self.learning_rate * gradient

    def newton_method(self, X, y):
        num_samples, num_features = X.shape
        num_classes = y.shape[1]
        self.theta = np.zeros((num_features, num_classes))
        for _ in range(self.epoch):
            h = self.softmax(np.dot(X, self.theta))
            gradient = np.dot(X.T, h - y) / num_samples
            Hessian = np.dot(X.T, np.diag(np.sum(h * (1 - h), axis=1))).dot(X) / num_samples
            self.theta -= np.dot(np.linalg.inv(Hessian), gradient)

    def fit(self, X, y, method):
        X = np.insert(X, 0, 1, axis=1)
        y = np.eye(int(np.max(y) + 1))[y.astype(int)]  # 转换y为one-hot编码
        if method == 'gradient_descent':
            self.gradient_descent(X, y)
        elif method == 'stochastic_gradient_descent':
            self.stochastic_gradient_descent(X, y)
        elif method == 'newton_method':
            self.newton_method(X, y)
        else:
            raise ValueError('Invalid method!')

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.argmax(self.softmax(np.dot(X, self.theta)), axis=1)


# 读取数据
X = np.loadtxt('../data/ex4x.dat')
y = np.loadtxt('../data/ex4y.dat')

# 初始化模型
model = SoftmaxRegression()

# 绘制数据点
plot_data(X, y)

# 使用梯度下降法训练模型
model.fit(X, y, method='gradient_descent')
print(f"梯度下降法得到的参数向量：{model.theta}")

# 绘制梯度下降法决策边界
plot_line_softmax(X, y, model, color="gray", name="GD")

# 使用随机梯度下降法训练模型
model.fit(X, y, method='stochastic_gradient_descent')
print(f"随机梯度下降法得到的参数向量：{model.theta}")

# 绘制随机梯度下降法决策边界
plot_line_softmax(X, y, model, color="pink", name="SGD")

# 使用牛顿法训练模型
model.fit(X, y, method='newton_method')
print(f"牛顿法得到的参数向量：{model.theta}")

# 绘制牛顿法决策边界
plot_line_softmax(X, y, model, color="green", name="Newton")

plt.legend()
plt.show()
