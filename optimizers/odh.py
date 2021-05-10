import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

class ODH:
    def __init__(self,
                 x0,
                 objFun,
                 gradFun,
                 A=None,
                 strategy='ODH1',
                 theta=0.01,
                 kappa=0.1,
                 maxIter=1e5,
                 tolerance=1e-3,
                 verbose=False):
        """
        Parameters:
        x0: np.array, starting point.
        objFun: The obective function to be minimized.
        gradFun: Function that calculates the gradient of objFun.
        A: objFun(x)=1/2*x^T*A*x + b^T*x
        strategy: {'ODH1', 'ODH2', 'adaptive1', 'adaptive2'}.
        maxIter: stop criteria, maximum iterations.
        tolerance: stop criteria.
        """
        gradx0 = gradFun(x0)
        self.objFun = objFun
        self.gradFun = gradFun
        self.x = [x0, x0+np.random.rand(len(x0)).reshape(-1,1)]
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

    def plot(self):
        import matplotlib.pyplot as plt
        iter = np.arange(0, len(self.gk_norm), 1)
        plt.plot(iter, self.gk_norm)
        plt.show()

    def optimize(self):
        gk_norm = np.linalg.norm(self.g[-1], ord=2)
        while True:
            # stop criterium
            if gk_norm<self.tolerance:
                print("gk_norm is small enough with value ", gk_norm) 
                break
            if self.iter>=self.maxIter:
                print('max number of iterations ', self.iter, ' is reached')
                break

            # calculate alpha
            alphak, status = self.calcAlpha(self.strategy)

            # stop criterium
            if status:
                print("xk do not change, optimize stopped.")
                print("Optimum: ", self.x[-1])
                break

            # update
            xk = self.x[-1] - 1./alphak*self.g[-1]
            try:
                gk = self.g[-1] - 1./alphak*self.A @ self.g[-1]
            except:
                gk = self.gradFun(xk)
            gk_norm = np.linalg.norm(self.g[-1], ord=2)
            self.iter+=1

            # recording
            self.alpha.append(alphak)
            self.g.append(gk)
            self.x.append(xk)
            self.gk_norm.append(gk_norm)

            if self.verbose:
                if self.iter%100==0:
                    print("Iter done:", self.iter)
                    print("alphak:", alphak, "gk_norm:", gk_norm, 'gk', gk,  "x:", xk)
                    # break

    def calcAlpha(self, strategy):
        assert strategy in ['ODH1', 'ODH2', 'adaptive1', 'adaptive2']
        status = 0
        # y[k] = g[k] - g[k-1]
        yk_old = self.g[-1] - self.g[-2]
        # s[k-1] = x[k] - x[k-1]
        sk_old = self.x[-1] - self.x[-2]

        if np.linalg.norm(sk_old, ord=2)<self.tolerance and np.linalg.norm(yk_old, ord=2)<self.tolerance:
            status = 1
        if strategy=='ODH1':
            return self.calcAlpha_ODH1(yk_old, sk_old), status
        elif strategy=='ODH2':
            return self.calcAlpha_ODH2(yk_old, sk_old), status
        else:
            alphak1 = self.calcAlpha_ODH1(yk_old, sk_old)
            alphak2 = self.calcAlpha_ODH2(yk_old, sk_old)

            if strategy=='adaptive1':
                if alphak1<=self.kappa*alphak2:
                    return alphak1, status
                else:
                    return alphak2, status
            elif strategy=='adaptive2':
                m = 100     ## TODO, this value should be more reasonable
                M = max(1, self.iter+1-m)
                if alphak1<=self.kappa*alphak2:
                    return np.min(np.append(self.alpha[-m:-1], alphak1)), status
                else:
                    return alphak2, status
        
    def calcAlpha_ODH1(self, yk_old, sk_old):
        Dk_old = (yk_old @ yk_old.transpose()) / (sk_old.transpose() @ yk_old)
        alphak = (self.theta * np.trace(Dk_old) + sk_old.transpose() @ yk_old) / (self.theta + sk_old.transpose() @ sk_old)
        return alphak

    def calcAlpha_ODH2(self, yk_old, sk_old):
        Hk_old = (sk_old @ sk_old.transpose()) / (sk_old.transpose() @ yk_old)
        alphak = (self.theta + (yk_old.transpose() @ yk_old)) / (self.theta * np.trace(Hk_old) + sk_old.transpose() @ yk_old)
        return alphak

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
