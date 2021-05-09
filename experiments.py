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
    b = A@np.random.rand(n).reshape(n, 1)

    def objFun(x):
        return 1/2 * x.transpose @ A @ x - x.transpose @ b

    def gradFun(x):
        return A @ x - b

    odhoptim = ODH(x0, objFun, gradFun, A, maxIter=5, verbose=True)
    odhoptim.optimize()

if __name__=='__main__':
    exp1()
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