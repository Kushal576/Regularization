#feature engineering



import numpy as np
import matplotlib.pyplot as plt
import math,copy


data=np.loadtxt("/home/acer/BCT/6th Semester/Regularization/overfitting1.txt",delimiter=",",dtype=float)
xtrain=data[:,:1]
ytrain=data[:,1:]
xfeature=np.c_[xtrain,xtrain**2]

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


def zscore_normalization(xreal):
  x=copy.deepcopy(xreal)
  mu=np.mean(x,axis=0)
  sigma=np.std(x,axis=0)
  x_normalized=(x-mu)/sigma
  return x_normalized,mu,sigma

w=np.zeros(xfeature.shape[1])
b=0
alpha=1.0e-1
x_normal,mu,sigma=zscore_normalization(xreal=xfeature)

print(f"Before starting:")
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(x_normal,ytrain,w,b)}")
print(f"After gradient descent")
w,b=gradient_descent(x_normal,ytrain,w,b,alpha,1000)
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(x_normal,ytrain,w,b)}")

xtest=np.linspace(1,20,15)
xtest=np.c_[xtest,xtest**2]
x,a,c=zscore_normalization(xtest)
y=[]
for i in range(x.shape[0]):
  y.append(predict(x[i],w,b))

fig,ax=plt.subplots()
ax.scatter(xtrain,ytrain)
ax.plot(xtest[:,:1],y)
plt.show()
