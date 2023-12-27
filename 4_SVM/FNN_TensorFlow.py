import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# 定义模型
def create_model(input, hidden_layers, hidden, output, hidden_activation, output_activation, loss, l_rate):
    # 创建一个 Sequential 模型，层按顺序堆叠
    model = tf.keras.Sequential()

    # 添加输入层，定义输入数据的形状（特征数量）
    model.add(tf.keras.layers.InputLayer(input_shape=(input,)))

    # 循环添加隐藏层
    for _ in range(hidden_layers):
        # 添加一个 Dense 层（全连接层），其中 n_hidden 是该层的神经元数量，activation是激活函数
        # 全连接层：它将上一层的所有神经元与当前层的所有神经元进行全连接
        model.add(tf.keras.layers.Dense(hidden, activation=hidden_activation))

    # 添加输出层
    model.add(tf.keras.layers.Dense(output, activation=output_activation))

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=l_rate),
                  loss=loss,
                  metrics=['accuracy'])
    return model


# 读取数据
X = np.loadtxt('../data/ex4x.dat')
y = np.loadtxt('../data/ex4y.dat')

# 对数据 x 进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建模型实例
n_input = X.shape[1]  # 输入层的神经元数量
n_hidden_layers = 5   # 隐藏层的数量
n_hidden = 4          # 隐藏层的神经元数量
n_output = 1          # 输出层的神经元数量
hidden_activation = 'relu'     # 隐藏层的激活函数
output_activation = 'sigmoid'  # 输出层的激活函数
loss = 'mean_squared_error'    # 损失函数
learning_rate = 0.001          # 学习率
epochs = 5000                  # 训练次数
model = create_model(n_input, n_hidden_layers, n_hidden, n_output, hidden_activation, output_activation, loss, learning_rate)

# 5倍交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
print("开始交叉验证")
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    # 测试模型，verbose=1 时输出测试过程
    _, accuracy = model.evaluate(X_test, y_test, verbose=1)
    accuracies.append(accuracy)

# 输出结果
print(f"每次交叉验证的准确率: {[f'{acc:.3%}' for acc in accuracies]}")
print(f"平均准确率: {np.mean(accuracies):.3%}")
