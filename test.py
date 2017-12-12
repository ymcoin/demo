import numpy as np
min = 0.5
max = 1.3
step = 0.02
l = 1+np.arange(min,max,step)
# print(np.random.randint(0, len(l)))
# print(np.random.rand(1)[0]+1)
print(l)
# print(l[np.random.randint(0, len(l))])
# print(l[np.random.randint(0, len(l))])

for i in range(0,10):
    a = l[np.random.randint(0, len(l))]
    b = l[np.random.randint(0, len(l))]
    if(i%2==0):
        while(a<b):
            a = l[np.random.randint(0, len(l))]
            b = l[np.random.randint(0, len(l))]
    else:
        while (a >= b):
            a = l[np.random.randint(0, len(l))]
            b = l[np.random.randint(0, len(l))]
    print (a,b)