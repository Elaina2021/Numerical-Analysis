import numpy as np

def matrix_multiple(A,B): # 矩阵乘法
	
	row_A = A.shape[0]
	if len(A.shape)==1:
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

A = np.array([
			[8,1,-2],
			[3,-10,1],
			[5,-2,20]
	])
x = np.array([
			[0],
			[0],
			[0]
	])
b = np.array([9,19,72])


def seidel(A,x,b,count=1,method='Seidel'):
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
	B_G = matrix_multiple(-(np.linalg.inv(D+L)),U)
	# print(B_G)
	d_G = matrix_multiple((np.linalg.inv(D+L)),b)

	if method == 'Jacobi':
		# print(1)
		x_prime = matrix_multiple(B_j,x)+d_j
	if method == 'Seidel':
		# print(2)
		x_prime = matrix_multiple(B_G,x)+d_G
	
	# 迭代误差
	epsilon = np.full((n,1),0.0000000001)
	if (abs(x_prime-x)<epsilon).all():
		print("result:",x)
		print("count:",count)
		return x
	else:
		x, count=x_prime, count+1
		if method == 'Jacobi':
			return seidel(A,x,b,count,method='Jacobi')
		if method == 'Seidel':
			return seidel(A,x,b,count,method='Seidel')

seidel(A,x,b)
print("Jacobi")
seidel(A,x,b,method='Jacobi')
AA = np.array([
				[6.,         3.25,       2.5025,    ],
 				[3.25,      2.5025,     2.090125,  ],
 				[2.5025,     2.090125,   1.82620625,],
 ])

x_0 = np.array([
			[0],
			[0],
			[0]
	])
YY = np.array(
	 [[10.942 ],
 [ 7.1855],
 [ 5.8566],]
	)

seidel(AA,x_0,YY,method='Seidel')
# seidel(AA,x_0,YY,method='Jacobi')