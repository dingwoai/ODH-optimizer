import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

class ODH:
    def __init__(self,
                 x0,
                 objFun,
                 gradFun,
                 A,
                 strategy='ODH1',
                 theta=0.01,
                 kappa=0.1,
                 maxIter=1e2,
                 tolerance=1e-6,
                 verbose=False):
        """
        Parameters:
        x0: np.array, starting point.
        objFun: The obective function to be minimized.
        gradFun: Function that calculates the gradient of objFun.
        strategy: {'ODE1', 'ODE2', 'adaptive'}.
        maxIter: stop criteria, maximum iterations.
        tolerance: stop criteria.
        """
        gradx0 = gradFun(x0)
        self.objFun = objFun
        self.gradFun = gradFun
        # self.x = [x0, self.backtracking(x0, gradx0)]
        self.x = [x0, x0+np.random.rand(len(x0))]
        self.alpha = []
        self.g = [gradx0, gradFun(self.x[-1])]
        self.gk_norm = []
        self.iter = 0
        self.A = A
        self.strategy = strategy
        self.theta = theta
        self.kappa = kappa
        self.maxIter = maxIter
        self.tolerance = tolerance
        self.verbose = verbose
        if self.verbose:
            print("Initial values: ", self.x, self.g)

    
    def backtracking(self, x0, g0):
        """ Suggested algorithm to avoid poor choices of 2 initial points """
        '''
        This function is modified from https://github.com/mesquita-daniel/StabBB
        '''
        alpha0 = 1 / np.linalg.norm(x0, np.inf)
        s0 = -alpha0 * g0
        x1 = x0 + s0
        while self.objFun(x1) > self.objFun(x0):
            s0 = s0 / 4
            x1 = x0 + s0
        return x1

    def optimize(self):
        gk_norm = np.linalg.norm(self.g[-1], ord=2)
        while gk_norm>self.tolerance and self.iter<self.maxIter:
            # calculate alpha
            alphak = self.calcAlpha(self.strategy)

            # update
            xk = self.x[-1] - 1./alphak*self.g[-1]
            gk = self.g[-1] - 1./alphak*self.A*self.g[-1]
            # gk = self.gradFun(xk)
            self.iter+=1

            # recording
            self.alpha.append(alphak)
            self.g.append(gk)
            self.x.append(xk)
            self.gk_norm.append(gk_norm)

            if self.verbose:
                if self.iter%1==0:
                    print("Iter:", self.iter, "alphak:", alphak, "gk_norm:", gk_norm, 'gk', gk,  "x:", xk)
                    # break

    def calcAlpha(self, strategy):
        assert strategy in ['ODH1', 'ODH2', 'adaptive1', 'adaptive2']
        # y[k] = g[k] - g[k-1]
        yk_old = self.g[-1] - self.g[-2]
        # s[k-1] = x[k] - x[k-1]
        sk_old = self.x[-1] - self.x[-2]
        if strategy=='ODH1':
            return self.calcAlpha_ODH1(yk_old, sk_old)
        elif strategy=='ODH2':
            return self.calcAlpha_ODH2(yk_old, sk_old)
        else:
            alphak1 = self.calcAlpha_ODH1(yk_old, sk_old)
            alphak2 = self.calcAlpha_ODH2(yk_old, sk_old)
            if strategy=='adaptive1':
                if alphak1<=self.kappa*alphak2:
                    return alphak1
                else:
                    return alphak2
            elif strategy=='adaptive2':
                if alphak1<=self.kappa*alphak2:
                    m = 100
                    return np.min([self.alpha[-m:-1], alphak1])
                else:
                    return alphak2
        
    def calcAlpha_ODH1(self, yk_old, sk_old):
        Dk_old = (yk_old @ yk_old.transpose()) / (sk_old.transpose() @ yk_old)
        # alphak = (self.theta * np.trace(Dk_old) + sk_old.transpose() @ yk_old) / (self.theta + sk_old.transpose() @ sk_old)
        alphak = (self.theta * Dk_old + sk_old.transpose() @ yk_old) / (self.theta + sk_old.transpose() @ sk_old)
        return alphak

    def calcAlpha_ODH2(self, yk_old, sk_old):
        Hk_old = (sk_old @ sk_old.transpose()) / (sk_old.transpose() @ yk_old)
        # alphak = (self.theta + (yk_old.transpose() @ yk_old)) / (self.theta * np.trace(Hk_old) + sk_old.transpose() @ yk_old)
        alphak = (self.theta + (yk_old.transpose() @ yk_old)) / (self.theta * Hk_old + sk_old.transpose() @ yk_old)
        return alphak

if __name__=='__main__':
    ## initialize
    n = 5
    # x0 = 10*np.random.rand(n).reshape(n,1)
    x0 = 2*np.array([1.3, 0.7, 0.8, 1.9, 1.2])
    objFun = rosen
    gradFun = rosen_der
    A = np.eye(n)
    odhoptim = ODH(x0, objFun, gradFun, A, verbose=True)
    odhoptim.optimize()

    # compare with scipy.minimize
    print('=================nelder-mead================')
    res = minimize(rosen, x0, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})
    
    print(res.x)
