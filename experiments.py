import time
import numpy as np
import argparse
from optimizers.odh import ODH
from optimizers.bb import BB
from utils.plot import *

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
    '''
    Experiment 1, Example 2
    '''
    n = 1000
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

    strategies=['ODH1', 'ODH2', 'AODH', 'AODHmin1']
    odhoptim = ODH(x0, objFun, gradFun, A, theta=n, strategies=strategies, maxIter=1e4, verbose=False, optim=xstar, expname='exp1')
    bboptim = BB(x0, objFun, gradFun, A, maxIter=1e4, verbose=False, optim=xstar, expname='exp1')
    # odhoptim.optimize()
    # odhoptim.plot()

def exp2():
    '''
    Experiment 2, A = diag(1, 2, ..., n), where n is the condition number of the Hessian of the function f(x)
    '''
    k = np.random.choice([1, 10, 25, 50, 100, 200]) ## memoryError for k>200
    k = 100
    n = 100*k
    print(n)
    A = np.diag(list(range(1, n+1)))
    x0 = np.zeros(n).reshape(n, 1)
    xstar = np.ones(n).reshape(n, 1)
    b = A@xstar

    def objFun(x):
        return 1/2 * x.transpose @ A @ x - x.transpose @ b

    def gradFun(x):
        return A @ x - b

    strategies=['ODH1', 'ODH2', 'AODH', 'AODHmin1']
    odhoptim = ODH(x0, objFun, gradFun, A, theta=n, strategies=strategies, maxIter=1e4, verbose=False, optim=xstar, expname='exp2')
    bboptim = BB(x0, objFun, gradFun, A, maxIter=1e4, verbose=False, optim=xstar, expname='exp2')

def exp4():
    '''
    Experiment 4, A is a tridiagonal matrix
    '''
    n = np.random.choice([500, 1000, 1500, 2000])
    print(n)
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

    strategies=['ODH1', 'ODH2', 'AODH', 'AODHmin1']
    odhoptim = ODH(x0, objFun, gradFun, A, theta=n, strategies=strategies, maxIter=3.5e4, verbose=False, optim=xstar, expname='exp4')
    strategies=['BB1', 'BB2', 'ABB', 'ABBmin1']
    bboptim = BB(x0, objFun, gradFun, A, strategies=strategies, maxIter=3.5e4, verbose=False, optim=xstar, expname='exp4')

def exp5():
    '''
    Experiment 5, compare the numerical performance of some methods in terms of the average number of iterations solving randomly generated sparse systems of equations
    '''
    ## https://github.com/zhh210/pypdas/blob/4c51ebdb5b1f407b80ba4a699d9bb1c9bf690dc1/pdas/randutil.py
    from cvxopt import matrix, spmatrix, normal, spdiag
    from math import pi, sin, cos, pow
    import random

    def sprandsym(size,cond=100,sp=0.5,vec=None):
        '''
        Generate random sparse positive definite matrix with specified 
        size, cond, sparsity. Implemented by random Jacobi rotation.
        '''
        def nnz(H):
            'Compute the number of non-zeros of matrix H'
            num = 0
            for i in H:
                if i != 0:
                    num += 1

            return num

        def rand_jacobi(H):
            '''
            Apply random Jacobi rotation on matrix H, preserve eigenvalues, 
            singular values, and symmetry
            '''
            (m,n) = H.size
            theta = random.uniform(-pi,pi)
            c = cos(theta)
            s = sin(theta)
            i = random.randint(0,m-1)
            j = i
            while j == i:
                j = random.randint(0,n-1)

            H[[i,j],:] =  matrix([[c,-s],[s,c]])*H[[i,j],:]
            H[:,[i,j]] =  H[:,[i,j]]*matrix([[c,s],[-s,c]])
            return H

        root = pow(cond,1.0/(size-1))
        if not vec:
            vec = [pow(root,i) for i in range(size)]
        H = spdiag(vec)
        dimension = size*size

        while nnz(H) < sp*dimension*0.95:
            H = rand_jacobi(H)
        return H

    n = 1000
    condA = 1000
    A = sprandsym(n, 1./condA, 0.8)  # in Matlab, A = sprandsym(n, 0.8, 1/condA, 1)
    A = np.array(matrix(A))
    x0 = (-10 + 20*np.random.rand(n, 1))
    xstar = (-10 + 20*np.random.rand(n, 1))
    b = A@xstar
    
    def objFun(x):
        return 1/2 * x.transpose @ A @ x - x.transpose @ b

    def gradFun(x):
        return A @ x - b

    t = np.random.choice([1e-1, 1e-3, 1e-6])
    print(n, t)
    strategies=['ODH1', 'ODH2', 'AODH', 'AODHmin1']
    odhoptim = ODH(x0, objFun, gradFun, A, theta=n, strategies=strategies, maxIter=2e4, tolerance=t, verbose=False, optim=xstar, expname='exp5')
    strategies=['BB1', 'BB2', 'ABB', 'ABBmin1']
    bboptim = BB(x0, objFun, gradFun, A, strategies=strategies, maxIter=2e4, tolerance=t, verbose=False, optim=xstar, expname='exp5')
    print(np.min(bboptim.alpha))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Select an experiment to run.')
    parser.add_argument('--exp', type=int, 
                        help='an integer for the experiment, {1:exp1, 2:exp2, 4:exp4, 5:exp5}')

    args = parser.parse_args()

    if args.exp==1:
        exp1()
        root_dir = './assets/'
        filenames = ['ODH1-exp1', 'ODH2-exp1', 'AODH-exp1', 'AODHmin1-exp1', 'BB1-exp1', 'BB2-exp1', 'ABB-exp1', 'ABBmin1-exp1']
        plot_from_npz(root_dir, filenames, savename='exp1.png')
    elif args.exp==2:
        exp2()
        root_dir = './assets/'
        filenames = ['ODH1-exp2', 'ODH2-exp2', 'AODH-exp2', 'AODHmin1-exp2', 'BB1-exp2', 'BB2-exp2', 'ABB-exp2', 'ABBmin1-exp2']
        plot_from_npz(root_dir, filenames, savename='exp2.png')
    elif args.exp==4:
        exp4()
        root_dir = './assets/'
        filenames = ['ODH1-exp4', 'ODH2-exp4', 'AODH-exp4', 'AODHmin1-exp4', 'BB1-exp4', 'BB2-exp4', 'ABB-exp4', 'ABBmin1-exp4']
        plot_from_npz(root_dir, filenames, savename='exp4.png')
    elif args.exp==5:
        exp5()
        root_dir = './assets/'
        filenames = ['ODH1-exp5', 'ODH2-exp5', 'AODH-exp5', 'AODHmin1-exp5', 'BB1-exp5', 'BB2-exp5', 'ABB-exp5', 'ABBmin1-exp5']
        plot_from_npz(root_dir, filenames, savename='exp5.png')
    else:
        print(f"Experiment {args.exp} is not implemented yet.")
    
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