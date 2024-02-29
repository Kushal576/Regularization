#feature engineering



import numpy as np
import matplotlib.pyplot as plt
import math,copy


data=np.loadtxt("/home/acer/BCT/6th Semester/Regularization/1.txt",delimiter=",",dtype=float)
xtrain=data[:,:1]
ytrain=data[:,1:]
xtrain=np.arange(0,20,1)
ytrain=xtrain*xtrain
xtrain = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
ytrain = np.array([0, 0, 0, 1, 1, 1])
xfeature=np.c_[xtrain]

def logistic_gradient(x,y,w,b):
  m,n=x.shape
  dj_dw=np.zeros((n,))
  dj_db=0.

  for i in range(m):
    z=np.dot(w,x[i])+b
    fwb=sigmoid(z)
    err=(fwb-y[i])
    for j in range(n):
      dj_dw[j]=dj_dw[j]+err*x[i,j]
    dj_db=dj_db+err
  dj_dw=dj_dw/m
  dj_db=dj_db/m

  return dj_dw,dj_db


def logistic_gradient_descent(x,y,w_in,b_in,alpha,iterations):
  w=copy.deepcopy(w_in)
  b=copy.deepcopy(b_in)

  for i in range(iterations):
    djw,djb=logistic_gradient(x,y,w,b)
    w=w-alpha*djw
    b=b-alpha*djb

  return w,b

def logistic_cost_function(x,y,w,b):
  m=x.shape[0]
  cost=0.0
  for i in range(m):
    z=np.dot(x[i],w)+b
    fwb=sigmoid(z)
    cost+=-y[i]*np.log(fwb)-(1-y[i])*np.log(1-fwb)

  cost=cost/(2*m)

  return cost

def predict(x,w,b):
  n=x.shape[0]
  yp=0
  for i in range(n):
    yp=yp+x[i]*w[i]
  yp=yp+b
  return yp 


def zscore_normalization(xreal):
  x=copy.deepcopy(xreal)
  mu=np.mean(x,axis=0)
  sigma=np.std(x,axis=0)
  x_normalized=(x-mu)/sigma
  return x_normalized,mu,sigma

def sigmoid(z):
  g=1/(1+np.exp(-z))
  return z

w=np.zeros(xfeature.shape[1])
b=0
alpha=1.0e-1
x_normal=xfeature

print(f"Before starting:")
print(f"(w,b)=({w},{b})")
print(f"Cost={logistic_cost_function(x_normal,ytrain,w,b)}")
print(f"After gradient descent")
w,b=logistic_gradient_descent(x_normal,ytrain,w,b,alpha,10000)
print(f"(w,b)=({w},{b})")
print(f"Cost={logistic_cost_function(x_normal,ytrain,w,b)}")

# xtest=np.linspace(1,20,15)
# xtest=np.c_[xtest,xtest**2]
# x,a,c=zscore_normalization(xtest)
# y=[]
# for i in range(x.shape[0]):
#   y.append(predict(x[i],w,b))

# y=x_normal@w+b

# fig,ax=plt.subplots()
# ax.scatter(xtrain,ytrain)
# ax.plot(xtrain,y)
# plt.show()
