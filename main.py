from kalman import *
from animation import Animator
steps = 100

R = 6400000
g = 9.81
class PlaneModel(Model):

    def Ak(self, k):
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


class FriendlyMissile(Model):

    def Ak(self, k):
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

S = Simulation(M2, Friendly)
anim = Animator(S, 1000)
anim.start_animation()




