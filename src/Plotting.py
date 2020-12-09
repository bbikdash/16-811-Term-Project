"""
@author: Bassam Bikdash, Ivan Cisneros
Plotting.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""

import numpy as np
import matplotlib.pyplot as plt


class Plotting:
    
    def plot_single_region(start, end, distortedAudio,
                           originalAudio, reducedAudio, cleanedAudio,
                           sampleRate, interp_points=6):
        # # Find regions of clipping
        # clippedRegLeft = detectClipping(distortedAudio[start:end, 0])
        
        # # Perform global amplitude reduction
        # reducedAudio = np.round(distortedAudio[start:end, 0] * SCALE)
        # print("clippedRegLeft: {}".format(clippedRegLeft))
        
        # # Perform cubic spline interpolation for every clipped region
        # cleanedAudio = np.zeros(reducedAudio.shape)
        # cleanedAudio = interpolate(clippedRegLeft, reducedAudio, interp_points)
        
        # cleanedAudio = np.round(cleanedAudio)
        # cleanedAudio = cleanedAudio.astype('int16')
        
        # Plotting
        length = distortedAudio.shape[0] / sampleRate
        time = np.linspace(0., length, distortedAudio.shape[0])
        plt.figure(1)
        plt.title('Original Audio and Distorted Audio in a Single Clipped Region')
        plt.plot(time[start:end], originalAudio[start:end, 0], 'b', label='Original (Left Channel)')
        plt.plot(time[start:end], distortedAudio[start:end, 0], 'r', label='Distorted (Left channel)')
        # plt.plot(time[1315:1325], distortedAudio[1315:1325,1], label="Right channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        
        plt.figure(2)
        plt.title('Distorted Audio and Cleaned Audio in a Single Clipped Region')
        plt.plot(time[start:end], reducedAudio[start:end, 0], 'r-+', label="Reduced/Distorted (Right Channel)")
        plt.plot(time[start:end], cleanedAudio[start:end, 0], 'g-*', label="Cleaned (Right Channel)")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()
        
        
        
    def generalPlot(originalAudio, distortedAudio, sampleRate):
    
        length = originalAudio.shape[0] / sampleRate
        time = np.linspace(0., length, originalAudio.shape[0])
        plt.figure(1)
        plt.title('Distorted Audio')
        # plt.plot(time, originalAudio[:, 0], 'r', label="Left channel")
        # plt.plot(time, originalAudio[:, 1], label="Right channel")
        plt.plot(time, distortedAudio[:, 0], 'r', label="Left channel")
        # plt.plot(time, distortedAudio[:, 1], label="Right channel")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        #
        # plt.figure(2)
        # plt.plot(time, reducedAudio[:, 0], label="Left channel")
        # plt.plot(time, reducedAudio[:, 1], label="Right channel")
        # plt.plot(time, cleanedAudio[:, 0], label="Left channel")
        # plt.plot(time, cleanedAudio[:, 1], label="Right channel")
        #
        # fig, axs = plt.subplots(2)
        # fig.suptitle('Distorted Audio')
        # axs[0].plot(time, distortedAudio[:, 0], label="Left channel")
        # axs[0].plot(time, distortedAudio[:, 1], label="Right channel")
        #
        # axs[1].plot(time, cleanedAudio[:, 0], label="Left channel")
        # axs[1].plot(time, cleanedAudio[:, 1], label="Right channel")
        #
        # plt.legend()
        # plt.xlabel("Time [s]")
        # plt.ylabel("Amplitude")
        
    
    def createSubplot(originalAudio, reducedAudio, reducedCleanAudio, sampleRate):
        # Plots 3 stacked graphs
        length = originalAudio.shape[0] / sampleRate
        time = np.linspace(0., length, originalAudio.shape[0])
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle("Original vs Clean")
        axs[0].plot(time, originalAudio[:, 0], color='r')
        axs[0].grid()
        axs[1].plot(time, reducedAudio[:, 0], color='b')
        axs[1].grid()
        axs[2].plot(time, reducedCleanAudio[:, 0], color='c')
        axs[2].grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        
        plt.show()
        
        