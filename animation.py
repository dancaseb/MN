import matplotlib.pyplot as plt
import matplotlib.animation as animation

from constants import friendly_missile_range
from kalman import Simulation
import numpy as np


class Animator:

    def __init__(self, simulation: Simulation, steps_number):
        self.simulation = simulation
        self.steps_number = steps_number
        self.trajectories_plots = []
        self.plots = []
        self.missile_plots = []
        self.predict_missile_plot = []
        self.friendly_missile_plot = []
        # Create the figure object and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.xlim_min, self.xlim_max = 0, 35000
        self.ylim_min, self.ylim_max = 0, 16000
        self.ax.set_xlim(self.xlim_min, self.xlim_max)
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        self.end_animation = False
        self.is_text = False
        # self.planets_animation = None

    def init_animation(self) -> list[plt.Axes.plot]:

        one_missile_trajectory, = self.ax.plot([], [], 'blue', ls='-')
        self.trajectories_plots.append(one_missile_trajectory)
        plot_missile, = self.ax.plot([], [], 'o', color='red', alpha=1)
        self.missile_plots.append(plot_missile)

        predict_missile_plot, = self.ax.plot([], [], 'o', color='gray', alpha=0.5)
        self.predict_missile_plot.append(predict_missile_plot)

        friendly_missile_plot, = self.ax.plot([], [], 'o', color='green', alpha=0.5)
        self.friendly_missile_plot.append(friendly_missile_plot)

        circle = plt.Circle(tuple(self.simulation.base), radius=friendly_missile_range, fill=False, linestyle='--')
        self.ax.add_patch(circle)

        # list of all plots. background, trajectories and planets
        self.plots = self.trajectories_plots + self.missile_plots + self.predict_missile_plot + self.friendly_missile_plot

        self.ax.legend([friendly_missile_plot, predict_missile_plot, plot_missile, one_missile_trajectory, circle],
                       ['protiraketova strela', 'predikovana poloha bomby', 'presna poloha bomby', 'predikovana trajektorie bomby', 'dostrel protiraketoveho systemu'])
        # return an iterable with plots to the animation function
        return self.plots

    def update(self, i):
        if self.simulation.was_successful is not True and self.simulation.system.x[1] >= 0:  # target was not hit and is still in the air

            self.simulation.run()
            # print(self.simulation.system.tracex[:, 0][-1])
            xlist = [x for x in self.simulation.filtr.trace_mu[0:i + 1, 0]]  # trace_mu[0:i + 1, 0]]
            # ylist = [y for y in self. ## trace_mu[0:i + 1, 1]]
            ylist = [y for y in self.simulation.filtr.trace_mu[0:i + 1, 1]]

            self.trajectories_plots[0].set_data(xlist, ylist)


            # x,y coordinates for the planets actual position.
            # At this position a circle representing the planet will be plotted
            x = self.simulation.system.tracex[i,0]
            y = self.simulation.system.tracex[i, 1]
            self.missile_plots[0].set_data(x, y)

            x = self.simulation.filtr.mu_p_ahead[0]
            y = self.simulation.filtr.mu_p_ahead[1]
            self.predict_missile_plot[0].set_data(x, y)

            x = self.simulation.friendly_system.x[0]
            y = self.simulation.friendly_system.x[1]
            self.friendly_missile_plot[0].set_data(x, y)
        if self.simulation.was_successful and not self.is_text:
            self.is_text = True
            plt.text(self.ax.get_xlim()[1]/2 - 5000, self.ax.get_ylim()[1]-1000, 'Zasah')

        self.plots = self.trajectories_plots + self.missile_plots + self.predict_missile_plot + self.friendly_missile_plot
        return self.plots

    def start_animation(self):
        """
        Function to start the animation. This is done by the FuncAnimation from matplotlib.Animation module.
        After exiting the animation is saved.
        :return:
        """

        self.planets_animation = animation.FuncAnimation(self.fig, self.update, frames = self.steps_number, init_func=self.init_animation,
                                                         interval=5, repeat=False)

        # Show the plot
        plt.show()
