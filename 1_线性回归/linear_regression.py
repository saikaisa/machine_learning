import numpy as np
import matplotlib.pyplot as plt


'''
====== LMS闭式解函数 ======
公式：theta = (X^T * X)^-1 * X^T * y
np.dot()：做矩阵乘法，X.dot(Y) 即为矩阵 X * Y
np.linalg.inv(a)：返回 a 的逆矩阵
'''
def closed_form(X, y):
    # 计算 X 的转置矩阵与 X 的矩阵的乘积
    XTX = X.T.dot(X)
    # 计算 (X^T * X) 的逆矩阵
    XTX_inv = np.linalg.inv(XTX)
    # 计算 X 的转置矩阵与 y 矩阵的乘积
    XTy = X.T.dot(y)
    # 将矩阵 XTX_inv 与 XTy 相乘，得到线性回归的闭式解
    theta = XTX_inv.dot(XTy)
    return theta


'''
====== 梯度下降函数 ======
损失函数（均方误差）：L = 1/2n * Σ(yi - (xi * θ))^2
对 θ 求偏导，得到梯度：G = 1/n * Σ2(xi * θ - yi) * xi
写成矩阵形式得：G = 2/n * X^T * (X * theta - y)
更新 θ 参数：θ' = θ - α * G  （α 为学习率）
'''
def gradient_descent(X, y, learning_rate, epoch):
    # 获取矩阵 X 的形状，num_samples 是样本数量，num_features 是特征数量
    num_samples, num_features = X.shape
    # 初始化参数向量 theta，长度为特征数量，初始值都为 0
    theta = np.zeros(num_features)
    # 运行 epoch 次
    for _ in range(epoch):
        # 计算梯度
        gradient = 2 / num_samples * X.T.dot(X.dot(theta) - y)
        # 更新 theta。新的 theta 等于旧的 theta 减去学习率乘以梯度。
        theta = theta - learning_rate * gradient
    return theta


'''
========= 假设线性回归函数为 y = wx + b =========
========= 定义 θ(theta) 为二维向量，θ = (w, b) =========
'''
# 数据。-2000 是为了让数据规模保持一致，之前由于 x y 相差太远导致报错了
x = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]) - 2000
y = np.array([2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900])
learning_rate = 0.01    # 梯度下降学习率（步长）
epoch = 1000    # 迭代次数

'''
====== 增加一列常数项 1，作为偏置项，类似于公式 y = wx + b 中的 b ======
np.shape() 返回 array 类型在某维度的长度，如 [4,3] 的二维数组 a 使用 a.shape[1] 返回 3
np.ones() 创建一个全为 1 的数组，括号里传入的参数是数组的形状
np.c_[] 沿着列（column）连接两个矩阵（要求两列相同）这里 X 连接完后变成一个二维数组，第一行全为 1，第二行为 x
'''
X = np.c_[np.ones(x.shape[0]), x]

# 使用闭式解求解
theta_closed = closed_form(X, y)
print(f"由闭式解得到的线性回归方程：y = {theta_closed[1]}(x-2000) + {theta_closed[0]}")

# 使用梯度下降求解
theta_gradient = gradient_descent(X, y, learning_rate, epoch)
print(f"由梯度下降得到的线性回归方程：y = {theta_gradient[1]}(x-2000) + {theta_gradient[0]}")

# 预测 2014 年的房价
x_pre = np.array([1, 2014 - 2000])
y_pre_closed = x_pre.dot(theta_closed)
y_pre_gradient = x_pre.dot(theta_gradient)

print(f"使用闭式解建立模型预测出的 2014 年平均房价: {y_pre_closed}")
print(f"使用梯度下降建立模型预测出的 2014 年平均房价: {y_pre_gradient}")

# 绘制数据点和拟合线
plt.scatter(x, y, color='blue', label='Raw data')
plt.plot(x, X.dot(theta_closed), 'r-', alpha=0.6, label='Closed form')
plt.plot(x, X.dot(theta_gradient), 'b--', alpha=0.6, label='Gradient descent')
plt.legend()
plt.show()
