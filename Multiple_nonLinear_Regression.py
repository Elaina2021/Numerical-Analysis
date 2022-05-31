# Time: 2022/5/30  16:28
import numpy as np
import pandas as pd


def h(theta, X): #定义假设函数 h(x) = theta_0 + theta_1*x_1 + theta_2*x_2
    return  np.dot(X, theta)

def costFunction(mytheta, X, y): #定义损失函数
    y_predict = h(mytheta, X)
    m = X.shape[0]
    return  float(    (1./(2*m))   *   np.dot((y_predict-y).T, (y_predict-y))     )

def gradientDescent(X, start_theta): #定义梯度下降函数 theta起点是[0,0,0]
    m = X.shape[0]
    theta = start_theta
    thetahistory = [] #用来存放theta值
    costhistory = [] #用来存放损失值
    for iter in range(max_iteration):
        tmptheta = theta
        costhistory.append(costFunction(theta, X,y))
        thetahistory.append(list(theta[:,0]))
        for j in range(len(tmptheta)):
            tmptheta[j] = theta[j] - (alpha/m)*np.sum((h(theta, X)-y)*np.array(X[:,j]).reshape(m, 1))
        theta = tmptheta
    return theta, thetahistory, costhistory # 返回最后一次的theta

def MSE_loss(Y_predict,Y):
    n = Y.shape[0]
    return (((Y_predict-Y)**2).sum())/n

def transform(X):
    t = X[:, 1] ** 2
    X[:, 1] = t
    return X


# data split
file = pd.read_csv('Student_Marks.csv')
df = pd.DataFrame(file)
train_data = df[:80]
test_data = df[80:]

train_dataset = train_data.values
train_dataset = transform(train_dataset)
mean = train_dataset.mean(axis=0)
std = train_dataset.std(axis=0)
print("mean:",mean,"std:",std)
data_normalized = (train_dataset - mean)/std #规范化处理
x = data_normalized[:,:2] #自变量
X = np.insert(x, 0,1 ,axis =1) #增加常数列
y = train_dataset[:,2].reshape([80,1]) #因变量, y在选取后需要reshape

# Training
print("Training...")
alpha = 0.1 #学习率
max_iteration = 300 #最大迭代次数
initial_theta = np.zeros((X.shape[1], 1)) #初始化theta值
theta, thetahistory, costhistory = gradientDescent(X, initial_theta) # 第一项是最后一次的theta
print("final loss on training data:",costhistory[-1])
print("theta:",theta.T)

# testing
print('Testing...')
test_dataset = test_data.values
test_dataset = transform(test_dataset)
test_data_normalized = (test_dataset - mean)/std # 标准化
X_test = test_data_normalized[:,:2]
X_test_input= np.insert(X_test, 0,1 ,axis=1) #增加常数列
Y_test = test_dataset[:,2].reshape([20,1])

predict = h(theta,X_test_input)
loss = MSE_loss(predict,Y_test)
print("MSE Loss on test data:",loss)
print("MAE Loss on test data:",np.sqrt(loss))


