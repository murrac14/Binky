import numpy

'''
# array takes a list to initialise

# 3x3 matrix of origin.
a_row1 = [1, 0, 0]
a_row2 = [0, 1, 0]
a_row3 = [0, 0, 1]
rows = [a_row1, a_row2, a_row3]
a = numpy.array(rows)

print(a)
print(a[1, 1])

# prints dimensions of matrix
print(a.shape)
print(a.shape[0])
print(a.shape[1])

b = numpy.array([[2.1, 1.6, 2.7],[3.3, 5.1, 2.8], [4.9, 2.6, 0.7]])
b[0, 0] = 1.3
print(b)
print()
list = b[0:2, 1:3]
print(list)
print()

# special matrices
c = numpy.array([[1, 6, 4], [0, 2, 4], [5, 1, 1]])

# identity contains floats. Takes dimensions as parameter
identity = numpy.eye(3)
# ones. must be past a tuple, hence extra brackets
ones = numpy.ones((2, 4))
print(ones)
# full matrix
full = numpy.full((5, 3), 4)
# random matrices
random = 5*numpy.random.random((3, 3))
print(random)
print()
# random
mean = numpy.mean(random)
standard_deviation = numpy.std(random)

# point-wise multiplication (think scalars times matrix)
a = numpy.array([[1, 2, 1], [5, 3, 1], [2, 1, 2]])
b = numpy.array([[2, 3, 1], [2, 4, 7], [3, 9, 5]])
print(a)
print()
print(b)
print()


c = a*b

# Matrix Products (proper linear algebra)
c = numpy.matmul(a, b)
#print(c)

# Transpose a matrix
#print(c.transpose())
#print(c.T)

# Dot Product aka Inner Product aka Matrix Multiplication
#a = numpy.arange(60.).reshape(3, 4, 5)
#b = numpy.arange(24.).reshape(4, 3, 2)
#c = numpy.dot(a, b)


# Tensor Dot Product
c = numpy.tensordot(a, b, axes=([1, 0], [0, 1]))
print(c)
print()

c = numpy.linalg.matrix_power(a, 2)
print(c)

# Kronecker Product
c = numpy.kron(a, b)

# Matrix Decompositions
a = numpy.array([[1, 5, 6],
                 [1, 8, 9],
                 [0, -1, 6]])

b = numpy.array([[2, 3, 1],
                 [2, 4, 7],
                 [3, 9, 5]])

d = numpy.array([[1 - 2j],
                [3 + 1j],
                [2 + 6j]])

print(a)
print()
print(b)
print()

# Cholesky Decomposition
c = numpy.linalg.cholesky(a)
print(c)

# Verify Result (Note that we will only get an approximation of a stored in b)
#b = numpy.matmul(c, c.transpose())
#print(b)


# QR Decomposition
(q, r) = numpy.linalg.qr(a)
#r = numpy.linalg.qr(a)[1]
print(q)
print(r)
print()
# Confirm result (again an approximation)
print(q*r)

# Eigenvalue Decomposition
a = numpy.array([[1, 5, 6],
                 [1, 8, 9],
                 [0, -1, 6]])

b = numpy.array([[2, 3, 1],
                 [2, 4, 7],
                 [3, 9, 5]])

d = numpy.array([[1 - 2j],
                [3 + 1j],
                [2 + 6j]])

print(a)
print()
print(b)
print()

# Get the eigenvector d, and eigenvalues matrix p
(d, p) = numpy.linalg.eig(a)
# d is a vector and not a matrix
print(d)
print()
print(p)
print()

# To verify result first convert vector d to matrix, then a = pdp^-1
# We will get complex result so extract the real and we get a very close approximation
D = numpy.diag(d)
P = numpy.matmul(numpy.matmul(p, D), numpy.linalg.inv(p))
P_real = numpy.real(P)

# Just eigenvalues
#print(numpy.linalg.eigvals(a))

# Singular Value Decomposition
U = numpy.linalg.svd(a)[0]
S = numpy.linalg.svd(a)[1]
V = numpy.linalg.svd(a)[2]

# Succinctly put:
(U, S, V) = numpy.linalg.svd(a)

print(U)
print(S)
print(V)

# Matrix Norms

# Usual norm, aka l2 Norm, aka Euclidean Norm
c = numpy.linalg.norm(a)
print(a, end="\n\n")

# Conditional Number, tells us how close to singularity the matrix is
# Singularity is how close the absolute value of the smallest eigenvalue is to 0
e = numpy.linalg.cond(b)
print(e, end="\n\n")

# Determinant: Tells us by how much a surface or area is scaled when multiplied by this matrix
# also used to find the inverse of a matrix
f = numpy.linalg.det(a)
print(f, end="\n\n")

# Rank of a matrix. The min(#rows, #columns)
# Aka number of linearly independent rows
# Aka number of linearly independent columns
g = numpy.linalg.matrix_rank(a)
print(g, end="\n\n")

# Trace of a matrix. The sum of the diagonal from 1,1 to n,m (in a nxm matrix)
h = numpy.trace(a)
print(h, end="\n\n")

# Solving Equations
# Ax = B
# Basically what is a post-multiplied by to give b
h = numpy.linalg.solve(a, b)
print(h, end="\n\n")

# Non-Invertible matrix,
# Use the psuedo inverse is used
# basically an approximation
s = numpy.array([[1, 2],
                 [2, 4]])

s_inv = numpy.linalg.pinv(s)
print(s_inv, end="\n\n")

x = numpy.matmul(s_inv, s)
print(x, end="\n\n")
'''