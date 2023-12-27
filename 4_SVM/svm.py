from libsvm.svmutil import svm_train, svm_predict, svm_problem, svm_parameter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np

# 读取数据
X = np.loadtxt('../data/ex4x.dat')
y = np.loadtxt('../data/ex4y.dat')

# 数据缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 交叉验证，返回平均正确率
def cross_validation(param):
    # 初始化5倍交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    # 5倍交叉验证
    for i, (train_index, test_index) in enumerate(kf.split(X_scaled)):
        # 初始化训练集和测试集
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 将数据转换为适合 SVM 包的格式
        data = svm_problem(y_train, X_train)
        # 训练模型
        model = svm_train(data, param)
        # 预测数据
        p_label, p_acc, p_val = svm_predict(y_test, X_test, model, '-q')
        accuracies.append(p_acc[0])

    return np.mean(accuracies)


# 存储最高准确率以及对应参数
best_acc_linear, best_C_linear = 0, 0
best_acc_poly, best_C_poly, best_degree_poly = 0, 0, 0
best_acc_rbf, best_C_rbf, best_gamma_rbf = 0, 0, 0

# 线性核函数
'''
np.logspace(m, n, num, base) 表示：
选取范围在 base^m ~ base^n 之间的 num 个数，
这些数在对数刻度上均匀分布，即每个数都是前一个数乘以 {base 的某个幂次} 的结果
'''
for C in np.logspace(-20, 1, num=10, base=2):
    '''
    -t 0 表示线性核函数，参数为 C
    -t 1 表示多项式核函数，参数为 C
    -t 2 表示径向基核函数，参数为 C 和 gamma
    '''
    # 设置 SVM 参数。-q: 抑制日志输出
    parameter = svm_parameter(f'-q -t 0 -c {C}')
    acc = cross_validation(parameter)
    if acc > best_acc_linear:
        best_acc_linear, best_C_linear = acc, C

print(f'线性核函数 - 最高准确率: {best_acc_linear}%, C: {best_C_linear}')

# 多项式核函数
for C in np.logspace(-10, 1, num=24, base=2):
    # 尝试不同的多项式度数
    for degree in range(1, 5):
        parameter = svm_parameter(f'-q -t 1 -c {C} -d {degree}')
        acc = cross_validation(parameter)
        if acc > best_acc_poly:
            best_acc_poly, best_C_poly, best_degree_poly = acc, C, degree

print(f'多项式核函数 - 最高准确率: {best_acc_poly}%, C: {best_C_poly}, 多项式度数: {best_degree_poly}')

# 径向基核函数
for C in np.logspace(0, 8, num=18, base=2):
    for gamma in np.logspace(-15, 3, num=10, base=2):
        parameter = svm_parameter(f'-q -t 2 -c {C} -g {gamma}')
        acc = cross_validation(parameter)
        if acc > best_acc_rbf:
            best_acc_rbf, best_C_rbf, best_gamma_rbf = acc, C, gamma

print(f'径向基核函数 - 最高准确率: {best_acc_rbf}%, C: {best_C_rbf}, Gamma: {best_gamma_rbf}')
