import numpy as np
import time
from scipy.optimize import minimize, rosen, rosen_der
from utils.plot import *

class BB:
    def __init__(self,
                 x0,
                 objFun,
                 gradFun,
                 A=None,
                 strategies=['BB1', 'BB2', 'ABB', 'ABBmin1'],
                #  theta=1,  ## not used in BB
                 kappa=0.5,
                 tau=0.65,
                 maxIter=1e5,
                 tolerance=1e-8,
                 verbose=False,
                 optim=0,
                 alphamin=1e-4,
                 alphamax=1e4,
                 expname='exp1'):
        """
        An unofficial implementation of Barzilar-Borwein (BB) method.
        Refer to the book 《最优化：建模、算法与理论》, page 229, 
        as well as paper "TWO NOVEL GRADIENT METHODS WITH OPTIMAL STEP SIZES".
        Parameters:
        x0: np.array, starting point.
        objFun: The obective function to be minimized.
        gradFun: Function that calculates the gradient of objFun.
        A: objFun(x)=1/2*x^T*A*x + b^T*x
        strategy: {'BB1', 'BB2', 'ABB', 'ABBmin1'}.
        maxIter: stop criteria, maximum iterations.
        tolerance: stop criteria.
        """
        gradx0 = gradFun(x0)
        self.objFun = objFun
        self.gradFun = gradFun
        self.x = [x0, x0+np.random.rand(len(x0)).reshape(-1,1)]
        self.alpha, self.alphak2 = [], []
        self.g = [gradx0, gradFun(self.x[-1])]
        self.gk_norm = []
        self.iter = 0
        self.A = A
        # self.strategy = strategy
        # self.theta = theta
        self.kappa = kappa
        self.tau = tau
        self.maxIter = maxIter
        self.tolerance = tolerance
        self.verbose = verbose
        self.optim = optim
        self.alphamin = alphamin
        self.alphamax = alphamax
        self.error = []
        if self.verbose:
            print("Initial values: ", self.x, self.g, 'Optima: ', self.optim)
        
        for self.strategy in strategies:
            self.reset()
            print("Strategy: ", self.strategy)
            start = time.time()
            self.optimize()
            print(f"Time cost {time.time()-start}, Iter {self.iter}, Error {self.error[-1]}, NormG is {self.gk_norm[-1]}")
            self.save_result(name=self.strategy+'-'+expname)

    def plot(self, save=True, filename='test.png'):
        # plot_single(self.gk_norm, 'gk_norm')
        plot_single(self.error, 'error', filename=self.strategy+'-error.png')
        # plot_multi([self.gk_norm, self.error], ['gk_norm', 'error'])

    def save_result(self, name):
        np.savez('./assets/'+name+'.npz', 
                 alpha=self.alpha,
                 g=self.g,
                 x=self.x,
                 gk_norm=self.gk_norm,
                 error=self.error)

    def reset(self):
        self.x = self.x[:2]
        self.alpha, self.alphak2 = [], []
        self.g = self.g[:2]
        self.gk_norm = []
        self.iter = 0
        self.error = []

    def optimize(self):
        '''
        Refer to Algorithm 6.2 in book.
        '''
        gk_norm = np.linalg.norm(self.g[-1], ord=2)
        while True:
            # stop criterium
            if gk_norm<self.tolerance:
                print(f"Iter {self.iter}, gk_norm is small enough with value {gk_norm}") 
                break
            if self.iter>=self.maxIter:
                print(f'max number of iterations {self.iter} is reached')
                break

            # calculate alpha
            alphak, status = self.calcAlpha(self.strategy)
            # correct/truncate alpha, for stable BB
            # alphak = np.min([np.max([alphak, self.alphamin]), self.alphamax])
            # if alphak <= 0:
            #     alphak = np.linalg.norm(self.sk_old) / np.linalg.norm(self.yk_old)
            # stop criterium
            if status:
                print(f"Iter {self.iter}, xk or gk do not change, optimize stopped. NormG is {gk_norm}")
                # print("Optimum: ", self.x[-1])
                break

            # update
            # xk = self.x[-1] - alphak*self.g[-1]  ## this is different with odh
            xk = self.x[-1] - 1./alphak*self.g[-1]
            gk = self.gradFun(xk)
            gk_norm = np.linalg.norm(self.g[-1], ord=2)
            error_norm = np.linalg.norm(xk-self.optim, ord=2)
            self.iter+=1

            # recording
            self.alpha.append(alphak)
            self.g.append(gk)
            self.x.append(xk)
            self.gk_norm.append(gk_norm)
            self.error.append(error_norm)

            if self.verbose:
                if self.iter%100==0:
                    print("Iter done:", self.iter)
                    print("alphak:", alphak, "gk_norm:", gk_norm)
                    # break

    def calcAlpha(self, strategy):
        assert strategy in ['BB1', 'BB2', 'ABB', 'ABBmin1']
        status = 0
        # y[k] = g[k] - g[k-1]
        yk_old = self.g[-1] - self.g[-2]
        self.yk_old = yk_old
        # s[k-1] = x[k] - x[k-1]
        sk_old = self.x[-1] - self.x[-2]
        self.sk_old = sk_old

        if np.linalg.norm(sk_old, ord=2)<self.tolerance and np.linalg.norm(yk_old, ord=2)<self.tolerance:
            ## stop if solution x AND gradient g do not change much
            status = 1
        if strategy=='BB1':
            return self.calcAlpha_BB1(yk_old, sk_old), status
        elif strategy=='BB2':
            return self.calcAlpha_BB2(yk_old, sk_old), status
        else:
            alphak1 = self.calcAlpha_BB1(yk_old, sk_old)
            alphak2 = self.calcAlpha_BB2(yk_old, sk_old)
            self.alphak2.append(alphak2)

            if strategy=='ABB':
                if alphak2<=self.kappa*alphak1:
                    return alphak2, status
                else:
                    return alphak1, status
            elif strategy=='ABBmin1':
                m = 9     ## this value is chosen following setting in paper
                M = max(1, self.iter+1-m)
                if alphak2<=self.tau*alphak1:
                    return np.min(self.alphak2[-M:-1]), status
                else:
                    return alphak1, status
        
    def calcAlpha_BB1(self, yk_old, sk_old):
        ## equation (6.2.9) in book
        # alphak = (sk_old.transpose() @ yk_old) / (yk_old.transpose() @ yk_old)
        ## equation (3.2) in paper
        alphak = (sk_old.transpose() @ yk_old) / (sk_old.transpose() @ sk_old)
        return alphak[0][0]

    def calcAlpha_BB2(self, yk_old, sk_old):
        ## equation (6.2.9) in book
        # alphak = (sk_old.transpose() @ sk_old) / (sk_old.transpose() @ yk_old)
        ## equation (3.2) in paper
        alphak = (yk_old.transpose() @ yk_old) / (sk_old.transpose() @ yk_old)
        return alphak[0][0]

if __name__=='__main__':
    ## initialize
    n = 5
    # x0 = 10*np.random.rand(n).reshape(n,1)
    x0 = 2*np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    objFun = rosen
    gradFun = rosen_der
    odhoptim = ODH(x0, objFun, gradFun, verbose=True)
    odhoptim.optimize()

    # compare with scipy.minimize
    print('=================nelder-mead================')
    res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})
    
    print(res.x)
