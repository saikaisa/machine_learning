import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_sgd(X, y, learning_rate=0.001, n_iterations=5000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for _ in range(n_iterations):
        # 随机选择一个数据点
        idx = np.random.randint(m)
        xi = X[idx, :]
        yi = y[idx]

        # 计算模型预测和实际值之间的差异
        z = np.dot(xi, weights) + bias
        prediction = sigmoid(z)
        error = prediction - yi

        # 计算梯度
        gradient_weights = xi * error
        gradient_bias = error

        # 更新权重和偏置
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return weights, bias


# 预测函数
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    prob = sigmoid(z)
    return [1 if i > 0.5 else 0 for i in prob]


# 加载数据
X = np.loadtxt('../data/ex4x.dat')
y = np.loadtxt('../data/ex4y.dat')

# 数据缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化5倍交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每次迭代的准确率
accuracies = []

for i, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    print(f"开始第 {i + 1} 轮交叉验证")
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 训练模型
    weights, bias = logistic_regression_sgd(X_train, y_train)

    # 在测试集上评估模型
    y_pred = predict(X_test, weights, bias)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)
    print(f"第 {i + 1} 轮交叉验证完成，准确率: {accuracy:.3%}")

# 计算平均准确率
print(f"平均准确率: {np.mean(accuracies):.3%}")
