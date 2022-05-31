# Time: 2022/5/24  11:41
import numpy as np

def h(theta, X): #定义假设函数 h(x) = theta_0 + theta_1*x_1 + theta_2*x_2
    return  np.dot(X, theta)

# best parameters
theta = np.array([[24.0915625],   [2.93019708], [13.26994405]])
mean = np.array([ 5.1125,    22.1317981, 24.0915625])
std = np.array([ 1.72477354, 19.78599923, 14.41254205])

x = np.ones([1,3])
a,b = eval(input('Input #courses and study time:'))

# normalization
x[0,1] = (a-mean[0])/std[0]
x[0,2] = (b**2-mean[1])/std[1]

# print(x)
predict = h(theta,x)
print("The predicted mark is:",predict)


