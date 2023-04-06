import numpy as np


class Model:
    def __init__(self):
        pass
    def set_D(self, D):
        self.D = D
    def Ak(self, k):
        pass
    def Bk(self, k):
        pass
    def Hk(self, k):
        pass
    def Qk(self, k):
        pass
    def Rk(self, k):
        pass
    def Uk(self, k):
        pass

class System:
    def __init__(self, model, x0):
        self.model = model
        self.x0 = x0
        self.reset()

    def reset(self):    
        self.k = 0
        self.x = self.x0
        self.y = None
        self.tracex = None
        self.tracey = None
        self.tracet = None
    
    def step(self):
        self.k = self.k+1
        A = self.model.Ak(self.k)
        B = self.model.Bk(self.k)
        H = self.model.Hk(self.k)
        Q = self.model.Qk(self.k)
        R = self.model.Rk(self.k)
        U = self.model.Uk(self.k)
        W = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        V = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        self.x = np.dot(A, self.x)+np.dot(B, U)+W
        self.y = np.dot(H, self.x)+V
        if self.k == 1:
            self.tracex = self.x[None, :]
            self.tracey = self.y[None, :]
            self.tracet = np.array([self.model.D])
        else:
            self.tracex = np.vstack([self.tracex, self.x])
            self.tracey = np.vstack([self.tracey, self.y])
            self.tracet = np.hstack([self.tracet, self.k*self.model.D])

    def run(self, n):
        for i in range(n):
            self.step()


class Filtr:
    def __init__(self, system, mu0, sigma0):
        self.system = system
        self.model = system.model
        self.mu = mu0
        self.sigma = sigma0
        self.k = 0
        self.trace_mu = None
        self.trace_mu_p = None
        self.trace_sigma = None
        self.trace_sigma_p = None
        
    def step(self):
        self.k = self.k+1
        A = self.model.Ak(self.k)
        B = self.model.Bk(self.k)
        H = self.model.Hk(self.k)
        Q = self.model.Qk(self.k)
        R = self.model.Rk(self.k)
        U = self.model.Uk(self.k)
        y = self.system.tracey[int(self.k)-1]
        
        self.mu_p = A.dot(self.mu) + B.dot(U)
        self.sigma_p = A.dot(self.sigma).dot(A.T)+Q
        self.K = self.sigma_p.dot(H.T).dot(np.linalg.inv(H.dot(self.sigma_p).dot(H.T)+R))
        self.mu = self.mu_p+self.K.dot(y-H.dot(self.mu_p))
        self.sigma = self.sigma_p-self.K.dot(H).dot(self.sigma_p)

        if self.k == 1:
            self.trace_mu = self.mu[None, :]
            self.trace_mu_p = self.mu_p[None, :]
            self.trace_sigma = self.sigma[None, :, :]
            self.trace_sigma_p = self.sigma_p[None, :, :]
        else:
            self.trace_mu = np.vstack([self.trace_mu, self.mu])
            self.trace_mu_p = np.vstack([self.trace_mu_p, self.mu_p])
            self.trace_sigma = np.vstack([self.trace_sigma, self.sigma[None, :,:]])
            self.trace_sigma_p = np.vstack([self.trace_sigma_p, self.sigma_p[None, :, :]])

    def run(self):
        for i in self.system.tracet:
            self.step()

