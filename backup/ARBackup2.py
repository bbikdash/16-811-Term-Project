# -*- coding: utf-8 -*-
"""
@author: Bassam Bikdash, Ivan Cisneros
Audio Reconstruction

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import CubicSpline

# These represent the bounds
MAX_CLIP = 32767
MIN_CLIP = -32768
THRESHOLD = 3
CLIP = MAX_CLIP - THRESHOLD
SCALE = 0.85

def detectClipping(data):
    regions = 0 # Number of regions that have clipping
    clipped_region = {} # Dictionary containing clipped regions (start and end indices)

    inMiddle = False
    bounds = np.zeros(2, dtype=int)

    for i in range(len(data)):

        if abs(data[i]) >= CLIP and inMiddle == False:
            # ASSERT: Found the start of a clipped region
            bounds[0] = i
            inMiddle = True
        elif abs(data[i]) < CLIP and inMiddle == True:
            # ASSERT: Found the end of a clipped region
            bounds[1] = i - 1
            inMiddle = False

            # Store the clipped region in the dictionary
            clipped_region[regions] = bounds
            regions += 1
            # Reset the bounds
            bounds = np.zeros(2, dtype=int)

    return clipped_region


def interpolate(regions, red):
    data = np.copy(red)
    for key in regions.keys():
        indices = regions[key]

        x = np.zeros(4, dtype=int)
        y = np.zeros(4, dtype=int)

        # Use points around the clipped region to create the interpolated spline
        x[0] = indices[0] - 1
        x[1] = indices[0]
        x[2] = indices[1]
        x[3] = indices[1] + 1

        y = data[x]
        cs = CubicSpline(x, y)

        for i in range(indices[0], indices[1] + 1):
            data[i] = cs(i)

    return data


if __name__ == "__main__":

    #   Load the audio as a waveform `y`
    #   Store the sampling rate as `sr
    filename = 'test_audio/Trumpet_Distorted1.wav'
    sampleRate, distortedAudio = wavfile.read(filename)
    distortedAudio = distortedAudio.astype('int32')

    filename = 'test_audio/Trumpet_Original.wav'
    sampleRate, originalAudio = wavfile.read(filename)
    originalAudio = originalAudio.astype('int32')
    # One of the first clipped regions in the distorted trumpet occurs at indices 1318 to 1319

    # Find regions of clipping
    clippedRegLeft = detectClipping(distortedAudio[:,0])
    clippedRegRight = detectClipping(distortedAudio[:,1])

    #%% Perform global amplitude reduction
    # Reduce by 10%
    reducedAudio = np.round(distortedAudio * SCALE)

    # Perform cubic spline interpolation for every clipped region
    cleanedAudio = np.zeros(reducedAudio.shape)
    cleanedAudio[:,0] = interpolate(clippedRegLeft, reducedAudio[:,0])
    cleanedAudio[:,1] = interpolate(clippedRegRight, reducedAudio[:,1])

    # Produce a new wavfile
    wavfile.write('Trumpet_Cleaned1', sampleRate, cleanedAudio)

    #%% Plotting
    length = originalAudio.shape[0] / sampleRate
    time = np.linspace(0., length, originalAudio.shape[0])
    plt.figure(1)
    plt.plot(time, originalAudio[:, 0], label="Left channel")
    plt.plot(time, originalAudio[:, 1], label="Right channel")
    plt.plot(time, distortedAudio[:, 0], label="Left channel")
    plt.plot(time, distortedAudio[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.figure(2)
    plt.plot(time, reducedAudio[:,0], label="Left channel")
    plt.plot(time, reducedAudio[:,1], label="Right channel")
    plt.plot(time, cleanedAudio[:,0], label="Left channel")
    plt.plot(time, cleanedAudio[:,1], label="Right channel")

    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
