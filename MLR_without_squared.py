# Time: 2022/5/30  16:28
import numpy as np
import pandas as pd


def h(theta, X): #定义假设函数 h(x) = theta_0 + theta_1*x_1**2 + theta_2*x_2
    return  np.dot(X, theta)

def costFunction(mytheta, X, y): #定义损失函数
    y_predict = h(mytheta, X)
    return  float(    (1./(2*m))   *   np.dot((y_predict-y).T, (y_predict-y))     )

def gradientDescent(X, start_theta = np.zeros([3,1])): #定义梯度下降函数 theta起点是[0,0,0]
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
    # print(n)
    return (((Y_predict-Y)**2).sum())/n


file = pd.read_csv('Student_Marks.csv')
df = pd.DataFrame(file)


train_data = df[:80]
test_data = df[80:]
data =train_data
data_normalized = (data - data.mean())/(data.std()) #规范化处理
cor_matrix = data_normalized.corr() #相关性矩阵
m=80
x = np.array(data_normalized[['number_courses','time_study']]).reshape(m,2) #自变量
X = np.insert(x, 0,1 ,axis =1) #增加常数列
y = np.array(data['Marks']).reshape(m, 1) #因变量
alpha = 0.1 #学习率
max_iteration = 300 #最大迭代次数



initial_theta = np.zeros((X.shape[1], 1)) #初始化theta值

theta, thetahistory, costhistory = gradientDescent(X, initial_theta) # 第一项是最后一次的theta

print(costhistory[-1])

# testing
print('Testing...')

test_data_normalized = (test_data - data.mean())/(data.std())
n =20
X_test = np.array(test_data_normalized[['number_courses','time_study']]).reshape([n,2])
X_test_input= np.insert(X_test, 0,1 ,axis =1) #增加常数列
Y_test = np.array(test_data['Marks']).reshape([n,1])
predict = h(theta,X_test_input)
loss = MSE_loss(predict,Y_test)
print("Loss on test data:",loss)

