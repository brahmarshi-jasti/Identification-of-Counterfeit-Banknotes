import pandas as pd
import numpy as np
import math

df = pd.io.parsers.read_csv(
	filepath_or_buffer='banknote_authentication.data',
	header=None,
	sep=',',
	)
df.columns = [0,1,2,3] + ['class label']
df.dropna(how="all", inplace=True) 

df.tail()

print (df)

X = df[[0,1,2,3]].values
y = df['class label'].values

np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(0,2):
	mean_vectors.append(np.mean(X[y==cl], axis=0))
	print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))

S_W = np.zeros((4,4))
for cl,mv in zip(range(0,2), mean_vectors):
	class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
	for row in X[y == cl]:
		row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
		class_sc_mat += (row-mv).dot((row-mv).T)
	S_W += class_sc_mat                             # sum class scatter matrices
print('within-class Scatter Matrix:\n', S_W)

overall_mean = np.mean(X, axis=0)

S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):  
	n = X[y==i,:].shape[0]
	print(n)
	mean_vec = mean_vec.reshape(4,1) # make column vector
	overall_mean = overall_mean.reshape(4,1) # make column vector
	S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('between-class Scatter Matrix:\n', S_B)

eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
	eigvec_sc = eig_vecs[:,i].reshape(4,1)   
	print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
	print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in decreasing order:\n')
for i in eig_pairs:
	print(i[0])

W = np.hstack((eig_pairs[0][1].reshape(4,1)))
print('Matrix W:\n', W.real)


def projection( A , B ) :
	return A.dot(B);

Wfinal = W.real
M1 = projection(mean_vectors[0],Wfinal)
M2 = projection(mean_vectors[1],Wfinal)

threshold = (M1 + M2) / 2.0

print (threshold)

tst = pd.io.parsers.read_csv(
	filepath_or_buffer='testing.data',
	header=None,
	sep=',',
	)

tst.columns = [0,1,2,3] + ['class label']
tst.dropna(how="all", inplace=True) 

tst.tail()

D = tst[[0,1,2,3]].values
c = tst['class label'].values

print (D)

testResult = []

for i in range(len(D)):
	temp =  projection(D[i],Wfinal)
	if temp < threshold :
		testResult.append(1)
	else :
		testResult.append(0)

for i in range(len(D)):
	print ('Expected : ', c[i] , ' Test Result : ', testResult[i])

