import numpy as np

# define the given data
x = np.random.randn(100, 1)
y = 2 * x + np.random.rand()
m = len(x)

def grandient_descent(x, y, w, b, learning_rate): 
    djdw = 0.0
    djdb = 0.0
    cost_function = 0.0
    for i in range(m):
        y_hat = x[i] * w + b
        y_org = y[i] * 1.0
        djdw = djdw + (y_hat - y_org) * y_hat
        djdb = djdb + (y_hat - y_org)
        cost_function = (y_hat - y_org) ** 2.0
        
    cost_function = cost_function / (2 * m)
    curr_w = w - learning_rate * (1 /m) * djdw
    curr_b = b - learning_rate * (1 /m) * djdb
    w = curr_w
    b = curr_b
    print(w, b, cost_function)
    return w, b
    
w = 0.0
b = 0.0
for i in range(1000):
    w, b = grandient_descent(x, y, w, b, 0.3)

print(w, b)
for i in range(m):
    cal_ans = w * x[i] + b
    org_ans = y[i]
    print(cal_ans, org_ans, (cal_ans - org_ans))
 
 
# Batch Gradient Descent Implementation:

# we need to find the value of w and b which will give us the accurate ans of y.

# in order to find the correct value of w and b we use gradient descent algo

# the goal of this algo is to decrease the value of cost_function.

# the more minimized the value of cost_function is the more acccurately we can find the value of y

# the formula for gradient descent is:

# y = f(x) = wx + b

# Cost_Function = J(w,b) = 1/2m * Σ[(f(x) - y) ^ 2]

# w = w - α * d(J(w,b))/dw 

# put the value of J(w,b) and find the derivate wrt w
# we get,

# w = w - α * Σ[(1/m * f(x) - y) * x] ------(1)

# similary,
# b = b - α * d(J(w,b))/db 
# b = b - α * Σ[(1/m * f(x) - y)] ------(2)

# (1) and (2) are the formulae which we use to find the value of w and b
# m = no of data points given
# Σ = summation from 0 to m
# α = learning rate
# y = output
# y_hat = the value which we cal using defined w and b
