import numpy as np

def matrix_multiple(A,B): # 矩阵乘法
	
	row_A = A.shape[0]
	if len(A.shape)==1: # 若A为向量
		pass
	col_A = A.shape[1]

	# B矩阵
	row_B = B.shape[0]
	if len(B.shape)==1: # 若B为向量
		col_B = 1
		C = np.empty((row_A,col_B))
		for i in range(row_A):
			C[i,0] = 0
			for a in range(col_A):
				C[i,0] += A[i,a]*B[a]	
	else: # B为矩阵
		col_B = B.shape[1]	
		C = np.empty((row_A,col_B))
		for i in range(row_A):
			for j in range(col_B):
				C[i,j] = 0
				for a in range(col_A):
					C[i,j] += A[i,a]*B[a,j]
	return C

def transpose(A): # 矩阵转置
	row_A = A.shape[0]
	col_A = A.shape[1]
	A_t = np.empty ((col_A,row_A)) # A transform
	for i in range(col_A):
		for j in range(row_A):
			A_t[i,j] = A[j,i]	
	return A_t


def inverse(A):
	'''
	LU分解法求矩阵的逆
	A = LU
	dooolittle decomposition
	'''
	row_A = A.shape[0]
	col_A = A.shape[1]

	# Doolittle decomposition
	L = np.eye(row_A)
	U = np.zeros((row_A,col_A))

	for j in range(col_A):
		U[0,j] = A[0,j]
	for i in range(col_A):
		L[i,0] = A[i,0]/U[0,0]
	# print("L is\n",L)
	# print("U is\n",U)	

	for k in range(1,row_A):
		for j in range(k,row_A):
			S = 0
			for t in range(k):
				S+=L[k,t]*U[t,j]
			U[k,j] = A[k,j]-S

		for i in range(k,row_A):
			S = 0
			for t in range(k):
				S+=L[i,t]*U[t,k]
			L[i,k] = (A[i,k]-S)/U[k,k]

	# print("L is\n",L)
	# print("U is\n",U)
	# print("right L inverse:",np.linalg.inv(L))
	# print("right U inverse:\n",np.linalg.inv(U))
	# 根据L，U求逆矩阵
	# L_inv
	L_inv = np.zeros((row_A,col_A))
	for j in range(row_A):
		for i in range(j,row_A):
			if i==j:
				L_inv[i,j]=1/L[i,j]
			elif i<j:
				L_inv[i,j]=0
			else:
				S = 0
				for k in range(j,i):
					S+=L[i,k]*L_inv[k,j]
				L_inv[i,j]=-(L_inv[j,j])*S

	# U_inv
	U_inv = np.zeros((row_A,col_A))
	for j in range(row_A):
		for i in range(j,-1,-1):
			# print(i)
			if i==j:
				U_inv[i,j]=1/U[i,j]
			elif i>j:
				U_inv[i,j]=0
			else:
				S = 0
				for k in range(i+1,j+1):
					S+=U[i,k]*U_inv[k,j]
				U_inv[i,j]=(-U_inv[i,i])*S

	# print("L_inv",L_inv)
	# print("U_inv\n",U_inv)
	A_inverse = matrix_multiple(U_inv,L_inv)
	return A_inverse


def solve_linear_eqs(A,x,b,count=1,method='Seidel'):
	'''
	求解线性方程组
	method: 
	1, Jacobi 
	2, Seidel
	'''
	row_A = A.shape[0]
	col_A = A.shape[1]
	n = x.shape[0]

	L = np.zeros((row_A,col_A))
	for i in range(row_A):
		for j in range(col_A):
			if j<i:
				L[i,j]=A[i,j]

	U = np.zeros((row_A,col_A))
	for i in range(row_A):
		for j in range(col_A):
			if j>i:
				U[i,j]=A[i,j]	

	# print(U)

	D = np.zeros((row_A,col_A))
	for i in range(row_A):
		for j in range(col_A):
			if j==i:
				D[i,j]=A[i,j]

	D_inverse = np.zeros((row_A,col_A))
	for i in range(row_A):
		for j in range(col_A):
			if j==i:
				D_inverse[i,j]=1/A[i,j]

	B_j = np.eye(row_A) - matrix_multiple(D_inverse,A)
	# print(B_j)
	d_j = matrix_multiple(D_inverse,b)
	# print(d_j)
	B_G = matrix_multiple(-inverse(D+L),U)
	# print(B_G)
	d_G = matrix_multiple(inverse(D+L),b)

	if method == 'Jacobi':
		# print(1)
		x_prime = matrix_multiple(B_j,x)+d_j
	if method == 'Seidel':
		# print(2)
		x_prime = matrix_multiple(B_G,x)+d_G
	
	# 迭代误差
	epsilon = np.full((n,1),0.00000000001)

	try:
		if (abs(x_prime-x)<epsilon).all():

			# print("result:",x)
			# print("count:",count)
			return x
		else:
			x, count=x_prime, count+1
			if method == 'Jacobi':
				return solve_linear_eqs(A,x,b,count,method='Jacobi')
			if method == 'Seidel':
				return solve_linear_eqs(A,x,b,count,method='Seidel')

	except: # 如果栈溢出，减小精度
		epsilon = np.full((n,1),0.00000001)
		if (abs(x_prime-x)<epsilon).all():

			# print("result:",x)
			# print("count:",count)
			return x
		else:
			x, count=x_prime, count+1
			if method == 'Jacobi':
				return solve_linear_eqs(A,x,b,count,method='Jacobi')
			if method == 'Seidel':
				return solve_linear_eqs(A,x,b,count,method='Seidel')		


def Least_square_method(X,Y,n): 
	'''
	最小二乘法
	'''
	A = np.empty ((len(X),n+1))  # 计算矩阵A
	for i in range(len(X)):
		for j in range(n+1):
			A[i,j] = X[i]**j

	A_t = transpose(A)
	AA = np.empty((n+1,n+1))
	AA = matrix_multiple(A_t,A) # 系数矩阵
	YY = matrix_multiple(A_t,Y) # 常数向量
	# print("corr matrix A is:\n",AA) # left part A of the normal system of equations系数矩阵
	# print("constant vector Y is:\n",YY) #常数向量
	x_0 = np.zeros((AA.shape[1],1)) # 迭代起点
	x = solve_linear_eqs(AA,x_0,YY) # 解线性方程组
	print("拟合多项式次数为:\n",n)
	print("拟合多项式系数为:\n",x)
	return x

	
X = np.array([0,0.2,0.5,0.7,0.85,1]) # 输入节点列表
Y = np.array([1,1.221,1.649,2.014,2.340,2.718]) # 函数值序列
print("X is:\n",X)
print("Y is:\n",Y)
a = Least_square_method(X,Y,n=4)