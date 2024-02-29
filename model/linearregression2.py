#multiple features



import numpy as np
import matplotlib.pyplot as plt
import math,copy


data=np.loadtxt("/home/acer/BCT/6th Semester/Regularization/data/data.txt",delimiter=",",dtype=float)
xtrain=data[:,:1]
ytrain=data[:,2:]
xtrain = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
ytrain = np.array([460, 232, 178])


def gradient(x,y,w,b):
  m,n=x.shape
  dj_dw=np.zeros((n,))
  dj_db=0.

  for i in range(m):
    fwb=np.dot(w,x[i])+b
    err=(fwb-y[i])
    for j in range(n):
      dj_dw[j]=dj_dw[j]+err*x[i,j]
    dj_db=dj_db+err
  dj_dw=dj_dw/m
  dj_db=dj_db/m

  return dj_dw,dj_db


def gradient_descent(x,y,w_in,b_in,alpha,iterations):
  w=copy.deepcopy(w_in)
  b=copy.deepcopy(b_in)

  for i in range(iterations):
    djw,djb=gradient(x,y,w,b)
    w=w-alpha*djw
    b=b-alpha*djb

  return w,b

def cost_function(x,y,w,b):
  m=x.shape[0]
  cost=0.0
  for i in range(m):
    fwb=np.dot(x[i],w)+b
    cost+=(fwb-y[i])**2

  cost=cost/(2*m)

  return cost

def predict(x,w,b):
  n=x.shape[0]
  yp=0
  for i in range(n):
    yp=yp+x[i]*w[i]
  yp=yp+b
  return yp 

w=np.zeros_like([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b=0
alpha=5.0e-7
print(f"Before starting:")
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(xtrain,ytrain,w,b)}")
print(f"After gradient descent")
w,b=gradient_descent(xtrain,ytrain,w,b,alpha,1000)
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(xtrain,ytrain,w,b)}")

