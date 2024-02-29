#L2-regularization in linear regression


import numpy as np
import matplotlib.pyplot as plt
import math,copy


data=np.loadtxt("/home/acer/BCT/6th Semester/Regularization/data/houses.txt",delimiter=",",dtype=float)
data=data[data[:,0].argsort()]
xtrain=data[:,:1]
ytrain=data[:,4:]
# xtrain=np.arange(0,20,1)
# ytrain=xtrain*xtrain
xfeature=np.c_[xtrain,xtrain**2,xtrain**3,xtrain**4,xtrain**5,xtrain**6,xtrain**7,xtrain**8,xtrain**9,xtrain**10,xtrain**11,xtrain**12]

def L2_gradient(x,y,w,b,lamda):
  m,n=x.shape
  dj_dw=np.zeros((n,))
  dj_db=0.

  for i in range(m):
    fwb=np.dot(w,x[i])+b
    err=(fwb-y[i])
    for j in range(n):
      dj_dw[j]=dj_dw[j]+err*x[i,j]
    dj_db=dj_db+err
  for i in range(n):
    dj_dw[i]=dj_dw[i]+w[i]*lamda
  dj_dw=dj_dw/m
  dj_db=dj_db/m

  return dj_dw,dj_db


def L2_gradient_descent(x,y,w_in,b_in,alpha,iterations,lamda):
  w=copy.deepcopy(w_in)
  b=copy.deepcopy(b_in)

  for i in range(iterations):
    djw,djb=L2_gradient(x,y,w,b,lamda)
    w=w-alpha*djw
    b=b-alpha*djb

  return w,b

def L2_cost_function(x,y,w,b,lamda):
  m=x.shape[0]
  cost=0.0
  reg_cost=0.0
  for i in range(m):
    fwb=np.dot(x[i],w)+b
    cost+=(fwb-y[i])**2

  for j in range(w.shape[0]):
    reg_cost=reg_cost+w[j]**2
  reg_cost=reg_cost/lamda

  cost=cost+reg_cost
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
lamda = 2
x_normal,mu,sigma=zscore_normalization(xreal=xfeature)

print(f"Before starting:")
print(f"(w,b)=({w},{b})")
print(f"Cost={L2_cost_function(x_normal,ytrain,w,b,lamda)}")
print(f"After gradient descent")
w,b=L2_gradient_descent(x_normal,ytrain,w,b,alpha,1000,lamda)
print(f"(w,b)=({w},{b})")
print(f"Cost={L2_cost_function(x_normal,ytrain,w,b,lamda)}")

xtest=np.linspace(1,20,15)
xtest=np.c_[xtest,xtest**2]
x,a,c=zscore_normalization(xtest)
y=[]
for i in range(x.shape[0]):
  y.append(predict(x[i],w,b))
y=x_normal@w + b
fig,ax=plt.subplots()
ax.scatter(xtrain,ytrain)
ax.plot(xtrain,y)
plt.show()
