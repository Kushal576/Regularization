import numpy as np
import matplotlib.pyplot as plt

# array=np.loadtxt("overfitting.txt",delimiter=",")
# fig,ax=plt.subplots()
# x=array[:,:1]
# y=array[:,1:]
# ax.scatter(x,y)
# plt.show()

array=np.linspace(1,10,20)
sq=array+10
array1=np.linspace(15,20,10)
sq1=array1+10
array=np.c_[array,sq]
array1=np.c_[array1,sq1]
array=np.concatenate((array1,array),axis=0)
print(array)
np.savetxt('/home/acer/BCT/6th Semester/Regularization/overfitting1.txt',array,delimiter=",")
