import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Animator:

    def __init__(self, tracex, trace_mu, steps_number):
        self.tracex = tracex
        self.trace_mu = trace_mu
        self.steps_number = steps_number
        self.trajectories_plots = []
        # self.planets_plot = []
        self.plots = []
        # self.background = []
        self.missile_plots = []
        # Create the figure object and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.xlim_min, self.xlim_max = 0, 30000
        self.ylim_min, self.ylim_max = 0, 16000
        self.ax.set_xlim(self.xlim_min, self.xlim_max)
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        # self.planets_animation = None

    def init_animation(self) -> list[plt.Axes.plot]:

        one_missile_trajectory, = self.ax.plot([], [], 'blue', ls='-')
        self.trajectories_plots.append(one_missile_trajectory)
        plot_missile, = self.ax.plot([], [], 'o', color='red', alpha=1)
        self.missile_plots.append(plot_missile)

        # list of all plots. background, trajectories and planets
        self.plots = self.trajectories_plots + self.missile_plots

        # return an iterable with plots to the animation function
        return self.plots

    def update(self, i):
        # print(i)
        xlist = [x for x in self.trace_mu[0:i + 1, 0]]
        ylist = [y for y in self.trace_mu[0:i + 1, 1]]

        self.trajectories_plots[0].set_data(xlist, ylist)


        # x,y coordinates for the planets actual position.
        # At this position a circle representing the planet will be plotted
        x = self.tracex[i,0]
        y = self.tracex[i, 1]
        self.missile_plots[0].set_data(x, y)

        self.plots = self.trajectories_plots + self.missile_plots

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
