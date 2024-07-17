w = [1,1]
b = -1
n = 0.1
epoch = 10
x1 = [0,0,1,1]
x2 = [0,1,0,1]

def andd(x1,x2,w,b,n,epoch):
    for i in epoch:
        for j in range(len(x1)):
            for k in range(len(w)):

                z = w[k]*x1[j] + w[k+1]*x2[j] + b
                if z > 0:
                    z = 1
                else:
                    z = 0
                
                

