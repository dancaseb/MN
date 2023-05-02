import numpy as np


class Model:
    """
    Model linearniho systemu s pozorovanim.
    Metody Ak, Bk, Hk, Qk, Rk, Uk budou v potomkovi prepsany pro konkretni model,
    vraci prislusne matice A, B, H, Q, K a ridici vstup U v case k. 
    """
    def __init__(self):
        pass

    def set_D(self, D):
        """
        Nastavi delku casoveho kroku D.
        """
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

    def Uk(self, k, **kwargs):
        pass

class System:
    """
    Linearni system pro simulaci dat.
    """

    def __init__(self, model, x0):
        """
        model (instance tridy Model): model systemu
        x0 (ndarray): pocatecni stav 
        """

        self.model = model  
        self.x0 = x0        
        self.reset()
 
    def reset(self):
        """
        pocatecni nastaveni pred simulaci systemu
        """

        self.k = 0           # cas
        self.x = self.x0     # stav v case k
        self.y = None        # pozorovani v case k
        self.tracex = None   # trajektorie stavu (x_0, ..., x_k)
        self.tracey = None   # trajektorie pozorovani (y_0, ..., y_k) 
        self.tracet = None   # trajektorie (spojitetho) casu (0*D, 1*D, ..., k*D) 
    
    def step(self):
        """
        krok simulace
        """

        self.k = self.k+1
        A = self.model.Ak(self.k)
        B = self.model.Bk(self.k)
        H = self.model.Hk(self.k)
        Q = self.model.Qk(self.k)
        R = self.model.Rk(self.k)
        U = self.model.Uk(self.k, y=self.x[1])  # vyska
        print(self.x[0], self.x[1])
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
        """
        provede n kroku simulace systemu pocinaje nasledujicim casem (k+1)
        """

        for i in range(n):
            self.step()


class Filtr:    
    """
    Kalmanuv filtr
    """

    def __init__(self, system, mu0, sigma0):
        """
        system (instance tridy System): system s ulozenou trajektorii pozorovani
        m0 (ndarray): str. hodnota pocatecniho odhadu
        sigma0 (ndarray): kovariancni matice pocatecniho odhadu
        """

        self.system = system        # system s ulozenymi trajektoriemi pozorovani a casu,
                                    # slouzi jako zdroj dat (y_1, ..., y_n)
                                    # instance tridy System
        self.model = system.model   # model systemu pro vypocet filtrace
                                    # instance tridy Model
        self.mu = mu0               # aktualni str. hodnota filtrace
        self.sigma = sigma0         # aktualni kovariancni matice filtrace
        self.k = 0                  # aktualni (diskretni) cas
        self.trace_mu = None        # posloupnost str. hodnot filtraci
        self.trace_mu_p = None      # posloupnost str. hodnot jednokrokovych predikci 
        self.trace_sigma = None     # posloupnost kovariancnich matic filtraci
        self.trace_sigma_p = None   # posloupnost kovariancnich matic filtraci
        
    def step(self):
        """
        krok Kalmanova filtru
        """

        self.k = self.k+1
        A = self.model.Ak(self.k)
        B = self.model.Bk(self.k)
        H = self.model.Hk(self.k)
        Q = self.model.Qk(self.k)
        R = self.model.Rk(self.k)
        U = self.model.Uk(self.k, y=self.mu[0])
        y = self.system.tracey[int(self.k)-1]
        
        # parametry jednokrokove predikce 
        self.mu_p = A.dot(self.mu) + B.dot(U)
        self.sigma_p = A.dot(self.sigma).dot(A.T)+Q
        # matice zisku
        self.K = self.sigma_p.dot(H.T).dot(np.linalg.inv(H.dot(self.sigma_p).dot(H.T)+R))
        # aktualizace parametru filtrace
        self.mu = self.mu_p+self.K.dot(y-H.dot(self.mu_p))
        self.sigma = self.sigma_p-self.K.dot(H).dot(self.sigma_p)

        # ulozeni parametru predikce a filtrace
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
        """
        vypocet filtraci pro celou posloupnost pozorovani 
        """

        for i in self.system.tracet:
            self.step()


class Animator:
    def __init__(self, trace):
        """
        trace: data ktera chceme vlozit do animace.
        """

    def animate(self):
        pass

