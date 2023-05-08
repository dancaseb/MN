from kalman import *
from animation import Animator


R = 6400000
g = 9.81
wind = 9
class PlaneModel(ModelProjekt):

    def Ak(self, k, **kwargs):
        return (np.array([[1, 0, self.D, 0],
                          [0, 1, 0, self.D],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]))

    def Bk(self, k):
        return (np.array([[0, 0],
                          [0, 0],
                          [self.D, 0],
                          [0, self.D]]))

    def Hk(self, k):
        return (np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]]))

    def Qk(self, k):
        return 0.01 * np.eye(4)

    def Rk(self, k):
        return 0.001 * np.eye(2)

    def Uk(self, k, **kwargs):
        y = kwargs['y']
        return np.array([0, -g * (1 - (2 * y / R))])


class FriendlyMissile(ModelProjekt):

    def Ak(self, k, **kwargs):
        return (np.array([[1, 0, self.D, 0],
                          [0, 1, 0, self.D],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]]))

    def Bk(self, k):
        return (np.array([[0, 0],
                          [0, 0],
                          [self.D, 0],
                          [0, self.D]]))

    def Hk(self, k):
        return (np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]]))

    def Qk(self, k):
        return 0.01 * np.eye(4)

    def Rk(self, k):
        return 0.001 * np.eye(2)

    def Uk(self, k, **kwargs):
        return np.array([0, 0])


M2 = PlaneModel()
M2.set_D(0.1)

Friendly = FriendlyMissile()
Friendly.set_D(0.1)

S = Simulation(M2, Friendly, np.array([0,0,0,0]),np.diag([1e6, 1e6, 1e6, 1e6]), np.array([0, 15000, 555, 0]),  np.array([30000, 0, 0, 0]))
anim = Animator(S, 1000)
anim.start_animation()





