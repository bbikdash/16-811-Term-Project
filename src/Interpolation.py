"""
@author: Bassam Bikdash, Ivan Cisneros
Interpolation.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""
import numpy as np
from scipy.interpolate import CubicSpline

class Interpolation:
    @staticmethod
    def cubic_spline_scipy(indices, data, numpoints=6):
        """
        By default, uses the Not-a-knot type of interpolation.
        """
        x = np.zeros(numpoints, dtype=int)
        # y = np.zeros(6, dtype=int)

        # Use points around the clipped region to create the interpolated spline
        mid = int(numpoints/2)
        for i in range(0, mid):
            x[i] = indices[0] - (mid-i-1)
            x[numpoints - (1 + i)] = indices[1] + (mid-i-1)
        # x[0] = indices[0] - 2
        # x[1] = indices[0] - 1
        # x[2] = indices[0]
        # x[3] = indices[1]
        # x[4] = indices[1] + 1
        # x[5] = indices[1] + 2
        y = data[x]

        return CubicSpline(x, y)

    @staticmethod
    def cubic_spline(indices, data):
        """
        A from scratch implementation of the Cubic Spline Interpolation.
        """
        # TODO
        pass


    @staticmethod
    def plot_waveform():
        pass