import numpy as np
import matplotlib.pyplot as plt
import copy
# x1=np.arange(0,20,1)
# x2=np.arange(30,40,1)
# x1=np.c_[x1,x1]
# x2=np.c_[x2,x2]

# np.savetxt("1.txt",x1,delimiter=",")

# np.savetxt("2.txt",x2,delimiter=",")


data=np.array([[9, 2, 3],
           [4, 5, 6],
           [7, 0, 5]])

data=data[data[:,0].argsort()]

print(data)


