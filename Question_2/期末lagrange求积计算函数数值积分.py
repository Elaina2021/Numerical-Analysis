import numpy as np
# from scipy import integrate

def Newton_Cotes_3(f,a,b):
	'''
	Simpson method
	'''
	return ((b-a)/6)*(f(a)+4*f((a+b)/2)+f(b))


def Integrate_Cotes(f,a=0,b=1,h=0.125,Type='simpson'):
	# 按h的间隔划分区间
	n = int((b-a)/h)
	x = np.zeros(n+1)
	for i in range(n+1):
		x[i] = a+i*h

	Sum = 0
	
	# 复化simpson求积
	if Type == 'simpson':
		for i in range(n):
			A = Newton_Cotes_3(f,x[i],x[i+1]) # simpson method
			Sum +=A 
	return Sum

def get_lag_base(X,a,b):
	n = len(X)
	A = np.zeros(n)
	# 构造基函数
	for i in range(n):

		def L_base(x):
			S = 1
			for j in range(n):
				if i!=j:
					S*=((x-X[j])/(X[i]-X[j]))
			return S
		
		A[i] = calculus(L_base,a,b) # 计算单个基函数的积分
	# print(A)
	return A

def calculus(f,a,b):
	'''
	使用复化Simpson公式计算积分
	'''
	I = Integrate_Cotes(f,a,b)
	# print(I)
	return I

def Lagrange_interpolation(X,Y,a,b):
	n = len(X)
	A = get_lag_base(X,a,b) # 计算插值基函数的积分 
	I = 0
	for i in range(n):
		I += A[i]*Y[i]
	return I


X = np.array([1/4,1/2,3/4])
Y = np.array([1/16,1/4,9/16])
a = 0
b = 1
I = Lagrange_interpolation(X,Y,a,b)
print(I)