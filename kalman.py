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
        self.mu_p_ahead = self.mu
        self.sigma_p_ahead = self.sigma
        
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
        U = self.model.Uk(self.k, y=self.mu[1])
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

    def run(self, n):
        """
        vypocet filtraci pro celou posloupnost pozorovani 
        """

        for i in range(n):
            self.step()

    def predict_step(self, k):
        A = self.model.Ak(k)
        B = self.model.Bk(k)
        Q = self.model.Qk(k)
        U = self.model.Uk(k, y=self.mu_p_ahead[1])

        self.mu_p_ahead = A.dot(self.mu_p_ahead) + B.dot(U)
        self.sigma_p_ahead = A.dot(self.sigma_p_ahead).dot(A.T)+Q

    def predict_ahead(self, n):
        self.mu_p_ahead = self.mu
        self.sigma_p_ahead = self.sigma
        for i in range(n):
            self.predict_step(self.k + i + 1)


class Simulation:
    def __init__(self, enemy_missile, friendly_missile):
        self.system = System(enemy_missile, np.array([0, 15000, 555, 0]))
        self.friendly_system = System(friendly_missile, np.array([30000, 0, 0, 0]))
        mu0 = np.array([0, 0, 0, 0])
        sigma0 = np.diag([1e6, 1e6, 1e6, 1e6])
        self.filtr = Filtr(self.system, mu0, sigma0)
        self.base = np.array([30000, 0])
        self.is_setup = False

    def run(self):
        self.system.run(1)
        self.filtr.run(1)

        if self.calculate_distance(self.base, np.array([self.filtr.mu_p_ahead[0], self.filtr.mu_p_ahead[1]])) > 10000:
            self.filtr.predict_ahead(200)
        else:
            self.run_friendly_system()
        if self.calculate_distance(np.array([self.friendly_system.x[0], self.friendly_system.x[1]]),
                                   np.array([self.system.x[0], self.system.x[1]])) < 50:
            print('hit')
    def run_friendly_system(self):
        if self.is_setup is False:
            self.is_setup = True
            direction_to_predicted_missile = np.array([self.filtr.mu_p_ahead[0], self.filtr.mu_p_ahead[1]]) - self.base
            friendly_speed = direction_to_predicted_missile/(200 * 0.1)
            print([self.filtr.mu_p_ahead[0], self.filtr.mu_p_ahead[1]])
            self.friendly_system.x0 = np.array([30000, 0, friendly_speed[0], friendly_speed[1]])
            self.friendly_system.reset()
            self.friendly_system.run(1)
        else:
            self.friendly_system.run(1)

    def calculate_distance(self, point1, point2):
        try:
            distance = np.linalg.norm(point1 - point2)
        except TypeError:
            return 150000
        return distance
