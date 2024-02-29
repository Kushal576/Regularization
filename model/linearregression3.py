#feature scaling



import numpy as np
import matplotlib.pyplot as plt
import math,copy


data=np.loadtxt("/home/acer/BCT/6th Semester/Regularization/data/houses.txt",delimiter=",",dtype=float)
xtrain=data[:,:4]
ytrain=data[:,4:]

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


def zscore_normalization(x):
  mu=np.mean(x,axis=0)
  sigma=np.std(x,axis=0)
  x_normalized=(x-mu)/sigma
  return x_normalized

w=np.zeros_like([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b=0
alpha=1.0e-1
xtrain=zscore_normalization(x=xtrain)

print(f"Before starting:")
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(xtrain,ytrain,w,b)}")
print(f"After gradient descent")
w,b=gradient_descent(xtrain,ytrain,w,b,alpha,1000)
print(f"(w,b)=({w},{b})")
print(f"Cost={cost_function(xtrain,ytrain,w,b)}")


y=[]
for i in range(xtrain.shape[0]):
  y.append(predict(xtrain[i],w,b))

fig,ax=plt.subplots()
# ax.scatter(xtrain[:,:1],ytrain)
ax.scatter(xtrain[:,:1],y)
plt.show()
