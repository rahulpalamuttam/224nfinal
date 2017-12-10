from sklearn.datasets import load_digits
import numpy as np

digits = load_digits()
digits.data.shape

X = np.vstack([digits.data[digits.target==i] for i in range(10)])
y = np.hstack([digits.target[digits.target==i] for i in range(10)])

#print X.shape
#print y.shape

#print digits.target[digits.target==0].shape
#print digits.target[digits.target==1].shape


print digits.data.shape
print digits.target.shape
