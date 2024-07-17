
def andd(x1,x2,y,w,b,n,epoch):
    for i in range(epoch) :
        for j in range(len(x1)):

                z = w[0]*x1[j] + w[1]*x2[j] + b
                if z >= 0:
                    y1 = 1
                else:
                    y1 = 0
                err = y[j]-y1
                if (err) != 0:
                     w[0] = w[0] + n*(err)*x1[j]
                     w[1] = w[1] + n*(err)*x2[j]
                     b = b + n*(err)
                print(w)
                
    return x1, x2, y, w, b

x1 = list(map(int, input("Enter 1st variable values: ").split()))
x2 = list(map(int, input("Enter 2nd variable values: ").split()))
y = list(map(int, input("Enter output variable values: ").split()))
epoch = int(input("Enter no. of epochs: "))
n = float(input("Enter the learning rate: "))
w = list(map(int, input("Enter initial weight: ").split()))
b = int(input("Enter the initial bias: "))

res = andd(x1,x2,y,w,b,n,epoch)
print(res)

                

