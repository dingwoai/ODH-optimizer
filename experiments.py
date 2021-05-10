import time
import numpy as np
from optimizers.odh import ODH

def exp0():
    '''
    A very simple problem for debug purpose.
    '''
    A = np.eye(1)
    b = np.eye(1)
    x0 = np.zeros(1)
    def objFun(x):
        return 1/2*x*x + x

    def gradFun(x):
        return x+1

    odhoptim = ODH(x0, objFun, gradFun, A, verbose=True)
    odhoptim.optimize()

def exp1():
    ## Experiment 1, Example 2
    n = 10
    lambda_min, lambda_max = 1, 1e3
    u_i = np.append(0.2*np.random.rand(int(np.floor(n/2))), np.ones(int(np.ceil(n/2)))-0.2*np.random.rand(int(np.ceil(n/2))))
    lambdas = lambda_min + (lambda_max - lambda_min)*u_i
    A = np.diag(lambdas)
    x0 = np.zeros(n).reshape(n,1)
    xstar = np.random.rand(n, 1)
    b = A@xstar

    def objFun(x):
        return 1/2 * x.transpose @ A @ x - x.transpose @ b

    def gradFun(x):
        return A @ x - b

    odhoptim = ODH(x0, objFun, gradFun, A, maxIter=1e5, verbose=True)
    odhoptim.optimize()
    odhoptim.plot()

def exp4():
    ## Experiment 4, A is a tridiagonal matrix
    n = np.random.choice([500, 1000, 1500, 2000])
    h = 11/n
    A = np.zeros([n, n])
    for i in range(0, n):
        A[i, i] = 2/h/h
        if i!=0:
            A[i, i-1] = -1/h/h
        if i!=n-1:
            A[i, i+1] = -1/h/h
    x0 = np.zeros(n).reshape(n, 1)
    xstar = (-10 + 20*np.random.rand(n, 1))
    b = A@xstar

    def objFun(x):
        return 1/2 * x.transpose @ A @ x - x.transpose @ b

    def gradFun(x):
        return A @ x - b

    start = time.time()
    odhoptim = ODH(x0, objFun, gradFun, A, strategy='ODH1', maxIter=1e4, verbose=False)
    odhoptim.optimize()
    # odhoptim.plot()
    print('n:', n, 'Nitr: ', odhoptim.iter, 'NrmG:', odhoptim.gk_norm[-1], 'Time(s):', time.time()-start)
    print('Optimum diff:', xstar-odhoptim.x[-1])

if __name__=='__main__':
    exp4()
    ## initialize
    # n = 5
    # # x0 = 10*np.random.rand(n).reshape(n,1)
    # x0 = 2*np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    # objFun = rosen
    # gradFun = rosen_der
    # A = np.eye(n)
    # odhoptim = ODH(x0, objFun, gradFun, A, verbose=True)
    # odhoptim.optimize()

    # # compare with scipy.minimize
    # print('=================nelder-mead================')
    # res = minimize(rosen, x0, method='nelder-mead',
    #            options={'xatol': 1e-8, 'disp': True})
    
    # print(res.x)