import numpy as np
import matplotlib.pyplot as plt
import math,copy


data=np.loadtxt("/home/acer/BCT/6th Semester/Regularization/data/data1.txt",delimiter=",",dtype=float)
xtrain=data[:,:1]
ytrain=data[:,2:]
#xtrain = np.array([1.0, 2.0])   #features
#ytrain = np.array([300.0, 500.0])   #target value



def gradient(x,y,w,b):
  m=x.shape[0]
  dj_dw=0.0
  dj_db=0.0

  for i in range(m):
    fwb=w*x[i]+b
    dj_dw=dj_dw+(fwb-y[i])*x[i]
    dj_db=dj_db+(fwb-y[i])
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
  cost=0
  for i in range(m):
    fwb=x[i]*w-b
    cost=cost+(fwb-y[i])**2

  cost=cost/(2*m)

  return cost

def zscore_normalization(xreal):
  x=copy.deepcopy(xreal)
  mu=np.mean(x,axis=0)
  sigma=np.std(x,axis=0)
  x_normalized=(x-mu)/sigma
  return x_normalized,mu,sigma


w=0
b=0
x_normal,mu,sigma=zscore_normalization(xtrain)
alpha=1.0e-1

print(f"Before starting:")
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(x_normal,ytrain,w,b)}")
print(f"After gradient descent")
w,b=gradient_descent(x_normal,ytrain,w,b,alpha,1000)
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(x_normal,ytrain,w,b)}")

x=np.linspace(-1,3,100)
y=w*x+b

x=x*sigma+mu

fig,ax=plt.subplots()
ax.scatter(xtrain,ytrain)
ax.plot(x,y)
ax.set_xlabel('House size')
ax.set_ylabel('Price')
plt.show()