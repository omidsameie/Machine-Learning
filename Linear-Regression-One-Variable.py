import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x = np.arange(1,11,1)
y0 = 2+ 4*x
y_rand = np.random.uniform(low=-0.1,high=0.1,size = len(x))
y = y0#+y_rand
x = x.reshape(len(x),1) #These are the essential steps
y =y.reshape(-1,1) # These are the essential steps
#lr1 = LinearRegression()
#lr1.fit(x,y)
#print(lr1.coef_)
#print(lr1.intercept_)
def hypothesis(x,a0,a1):
    return a0+a1*x
def linear_regression(x, y,a0,a1, learning_rate,iterate):
    m = len(x) # length of inputs
    J = []
    for i in range(iterate):
        #print(i)
        temp0 = a0
        temp1 = a1
        h = hypothesis(x,temp0,temp1)
        J.append(np.sum((h-y)**2)/(2*m))
        da0 =learning_rate*np.sum(h-y)/m
        da1 =learning_rate*np.sum((h-y)*x)/m
        a0 = temp0-da0
        a1 = temp1 -da1
        #print(y_theory)
        #pdb.set_trace()
    return (a0,a1,J)
(a0,a1,J) = linear_regression(x,y,2.,2., 0.00001,10000)
