from matplotlib.pylab import plot, show, scatter

def main():
    X = [1,2,3,4,5,6,7,8]
    Y = [3,6,5,9,10,14,18,20]
    N = 8
    
    w, b = do_linear_regression(X, Y, N)
    print(w,b)
    y = [(w * number + b) for number in X]
    scatter(X,Y)
    plot(X,y)
    show()

def do_linear_regression(X, Y, N, rate=0.001, epochs=1000):
    w_0 = 1.0
    b_0 = 1.0
    for t in range(epochs):
        w_grad, b_grad = gradient_descendant(X,Y,N,w_0,b_0)
        w_0 -= rate * w_grad
        b_0 -= rate * b_grad
    return w_0, b_0

def gradient_descendant(X,Y,N,w_0,b_0):
    w_grad = 0
    b_grad = 0
    for i in range(N):
        w_grad += X[i] * (X[i] * w_0 + b_0 -Y[i])
        b_grad += (X[i] * w_0 + b_0 - Y[i])
    return w_grad, b_grad
    

if __name__ == "__main__":
    main()
