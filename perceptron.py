def summ(w, x1, x2, b):
    z_list = []
    for i in range(len(x1)):
        z = w[0] * x1[i] + w[1] * x2[i] + b
        z_list.append(z)
    return z_list

def sigmoid(z):
    if z >= 0:
        y1 = 1
    else:
        y1 = 0
    return y1

def updateweight(y, y1, w, n, x1, x2, b, i):
    err = y[i] - y1
    w[0] = w[0] + n * err * x1[i]
    w[1] = w[1] + n * err * x2[i]
    b = b + n * err
    return w, b

def andd(x1, x2, y, w, b, n, epoch):
    for i in range(epoch):
        for j in range(len(x1)):
            z = w[0] * x1[j] + w[1] * x2[j] + b
            y1 = sigmoid(z)
            w, b = updateweight(y, y1, w, n, x1, x2, b, j)
            print(f'Epoch {i+1}, Sample {j+1}: Weights: {w}, Bias: {b}')
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

                

