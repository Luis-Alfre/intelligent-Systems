import pandas as pd
import random
import matplotlib.pyplot as plt

def gen_data(n, bias, varianza):
    x = []
    y = []
    for i in range(0, n):
        x.append(i)
        y.append((i + bias) + random.uniform(0, 1) * varianza)
    x = [3060,1600,2000,1300,2000,1956,2400,1200,1800,1248,2025,1800,1100,3000,2000]
    y = [179000,126500,134500,125000,142000,164000,146000,129000,135000,118500,160000,152000,122500,220000,141000]
    return x, y

x,y = gen_data(100, 25, 50)

plt.scatter(x, y)
plt.show()

def coste(x, y, a, b):
    m = len(x)
    error = 0.0
    for i in range(m):
        hipotesis = a+b*x[i]
        error +=  (y[i] - hipotesis) ** 2
    return error / (2*m)


def descenso_gradiente(x, y, a, b, alpha, epochs):
    m = len(x)
    hist_coste = []
    for ep in range(epochs):
        b_deriv = 0
        a_deriv = 0
        for i in range(m):
            hipotesis = a+b*x[i]
            a_deriv += hipotesis - y[i]
            b_deriv += (hipotesis - y[i]) * x[i]
            hist_coste.append(coste(x, y, a, b))
        a -= (a_deriv / m) * alpha
        b -= (b_deriv / m) * alpha
        
    return a, b, hist_coste


a=100000
b=20
alpha = 0.0001
iters = 1
a,b, hist_coste = descenso_gradiente(x, y, a, b, alpha, iters)
print(a)
print(b)
plt.scatter(x, y)
pred_x = [0, max(x)]
pred_y = [a+b*0, a+b*max(x)]
plt.title('100000 Iteraciones')
plt.axis([0, 100, 0, 200]) 
plt.plot(pred_x, pred_y, "r")
plt.show()



def pred(a, b, val):
    return a+b*val

pred (a, b, 50)

x_base = range(len(hist_coste))
plt.plot(x_base[1000:], hist_coste[1000:])
plt.show()


