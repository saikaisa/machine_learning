import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Sigmoid激活函数的导数"""
    return x * (1 - x)


def initialize_network(n_inputs, n_hidden, n_outputs):
    """初始化神经网络"""
    network = {
        # 每个隐藏元和输出元，通过定义各成分间的关系，隐式地存储在这个字典里
        'hidden': {
            # 返回一个形状为 (n_inputs, n_hidden) 的二维数组
            # 代表每个输入层神经元与每个隐藏层神经元都有一个连接权重 Vij (i,j分别是输入元和隐藏元的编号)
            'weights': np.random.randn(n_inputs, n_hidden),
            # 形状为 (n_hidden,) 的一维数组，对应于每个隐藏层神经元的偏置
            'bias': np.zeros(n_hidden)
        },
        'output': {
            # 代表每个隐藏层神经元与每个输出层神经元都有一个连接权重 Wij
            'weights': np.random.randn(n_hidden, n_outputs),
            # 对应于每个输出层神经元的偏置
            'bias': np.zeros(n_outputs)
        }
    }
    return network


def forward_propagate(network, x):
    """前向传播"""
    # x(2,): 上个神经元的输出  z(5,): 中间结果/净输入到当前神经元  y(5,): 作为当前神经元的输出
    hidden_z = np.dot(x, network['hidden']['weights']) + network['hidden']['bias']
    hidden_y = sigmoid(hidden_z)

    # x(5,): 上个神经元的输出  z(5,): 中间结果/净输入到当前神经元  y(1,): 作为当前神经元的输出
    output_x = hidden_y  # 这一步只是为了看起来更清晰
    output_z = np.dot(output_x, network['output']['weights']) + network['output']['bias']
    output_y = sigmoid(output_z)

    return hidden_y, output_y


def backward_propagate(network, l_rate, inputs, hidden_y, expected_output_y, output_y):
    """BP算法"""
    # 计算输出层的误差和梯度
    output_error = (output_y - expected_output_y) * sigmoid_derivative(output_y)

    # 计算隐藏层的误差和梯度
    hidden_error = np.dot(output_error, network['output']['weights'].T) * sigmoid_derivative(hidden_y)

    # 更新输出层权重和偏置
    # output_error(1,)  hidden_y(5,)  -->  w(5,1)
    network['output']['weights'] -= np.dot(hidden_y.reshape(-1, 1), output_error.reshape(1, -1)) * l_rate
    network['output']['bias'] -= output_error * l_rate

    # 更新隐藏层权重和偏置
    network['hidden']['weights'] -= np.dot(inputs.T.reshape(-1, 1), hidden_error.reshape(1, -1)) * l_rate
    network['hidden']['bias'] -= hidden_error * l_rate


def train_network(network, X_train, y_train, l_rate, epochs):
    for epoch in range(epochs):
        for x, y in zip(X_train, y_train):
            # 前向传播
            hidden_y, output_y = forward_propagate(network, x)

            # 反向传播和更新权重（BP算法）
            backward_propagate(network, l_rate, x, hidden_y, y, output_y)


def cross_validation(sample_X, sample_y, n_folds, l_rate, epochs):
    kf = KFold(n_splits=n_folds)
    accuracies = []

    for i, (train_index, test_index) in enumerate(kf.split(sample_X)):
        print(f"开始第 {i + 1} 轮交叉验证")
        # print("y的类型：", type(sample_y))
        # print("y的形状：", sample_y.shape)
        # print(f"训练集：{train_index}")
        # print(f"测试集：{test_index}")

        X_train, X_test = sample_X[train_index], sample_X[test_index]
        y_train, y_test = sample_y[train_index], sample_y[test_index]

        # print("训练集和测试集的形状：", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        # 为每次分割初始化网络
        trained_network = initialize_network(n_inputs, n_hidden, n_outputs)
        train_network(trained_network, X_train, y_train, l_rate, epochs)

        # 评估模型
        correct_predictions = 0
        for x, y in zip(X_test, y_test):
            _, output = forward_propagate(trained_network, x)
            # print(f"输出：{output}")
            prediction = 1 if output > 0.5 else 0
            if prediction == y:
                correct_predictions += 1

        accuracy = correct_predictions / len(X_test)
        accuracies.append(accuracy)
        print(f"第 {i + 1} 轮交叉验证完成，准确率: {accuracy:.3%}")

    return accuracies


# 初始化网络
n_inputs = 2  # 两次考试的成绩
n_hidden = 5  # 隐藏层神经元数量（可以调整）
n_outputs = 1  # 输出层神经元数量

# 读取数据
sample_X = np.loadtxt('../data/ex4x.dat')
sample_y = np.loadtxt('../data/ex4y.dat')
# 对数据 x 进行标准化
scaler = StandardScaler()
sample_X = scaler.fit_transform(sample_X)

# 设置训练参数
l_rate = 0.01
epochs = 5000
n_folds = 5

# 进行5倍交叉验证
accuracies = cross_validation(sample_X, sample_y, n_folds, l_rate, epochs)

# 打印每次交叉验证的准确率和平均准确率
print(f"每次交叉验证的准确率: {[f'{acc:.3%}' for acc in accuracies]}")
print(f"平均准确率: {np.mean(accuracies):.3%}")
