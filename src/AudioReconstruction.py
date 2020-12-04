"""
@author: Bassam Bikdash, Ivan Cisneros
AudioReconstruction.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import Interpolation
import Comparison

# These represent the bounds
MAX_CLIP = 32767
MIN_CLIP = -32768
THRESHOLD = 5900  # originally 3
CLIP = MAX_CLIP - THRESHOLD

def detectClipping(data):
    clipped_region = []  # List containing clipped regions (start and end indices)

    inMiddle = False
    bounds = np.zeros(2, dtype=int)

    for i in range(len(data)):

        if abs(data[i]) >= CLIP and inMiddle == False:
            # ASSERT: Found the start of a clipped region
            bounds[0] = i - 1  # Off by one error found here
            inMiddle = True
        elif abs(data[i]) < CLIP and inMiddle == True:
            # ASSERT: Found the end of a clipped region
            bounds[1] = i - 1
            inMiddle = False

            # Store the clipped region in the list
            clipped_region.append(bounds)
            # Reset the bounds
            bounds = np.zeros(2, dtype=int)

    return clipped_region


def interpolate(regions, red, numpoints=6):
    data = np.copy(red)
    num_regions = len(regions)
    for i in range(0, num_regions):
        indices = regions[i]

        cs = Interpolation.cubic_spline_scipy(indices, data, numpoints)

        for i in range(indices[0], indices[1] + 1):
            data[i] = cs(i)

    return data



def plot_single_region(start, end, distortedAudio, originalAudio, sampleRate, interp_points=6):
    # Find regions of clipping
    clippedRegLeft = detectClipping(distortedAudio[start:end, 0])
    # Perform global amplitude reduction
    reducedAudio = np.round(distortedAudio[start:end, 0] * SCALE)
    print("clippedRegLeft: {}".format(clippedRegLeft))

    # http://fourier.eng.hmc.edu/e176/lectures/ch7/node6.html

    # Perform cubic spline interpolation for every clipped region
    cleanedAudio = np.zeros(reducedAudio.shape)
    cleanedAudio = interpolate(clippedRegLeft, reducedAudio, interp_points)

    cleanedAudio = np.round(cleanedAudio)
    cleanedAudio = cleanedAudio.astype('int16')

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
    plt.plot(time[start:end], reducedAudio, 'r-+', label="Reduced/Distorted (Right Channel)")
    plt.plot(time[start:end], cleanedAudio, 'g-*', label="Cleaned (Right Channel)")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


if __name__ == "__main__":
    """
    For all two channel matrices, we will adopt the convention:
        - left channel: audio[:,0]
        - right channel: audio[:,1]
    """
    # Essential ------------------------------------------------------------------------------------------
    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr
    filename = 'test_audio/Trumpet_Distorted1.wav'
    sampleRate, distortedAudio = wavfile.read(filename)
    distortedAudio = distortedAudio.astype('int32')
    # print("distortedAudio.shape: {}".format(distortedAudio.shape))

    filename = 'test_audio/Trumpet_Original.wav'
    sampleRate, originalAudio = wavfile.read(filename)
    originalAudio = originalAudio.astype('int32')
    # One of the first clipped regions in the distorted trumpet occurs at indices 1317 to 1320
    # ----------------------------------------------------------------------------------------------------


    # Comparing Cleaning Quality through FFT -----------------------------------------------------------
    # FFT histograms
    originalAudio_hist = Comparison.freq_hist(originalAudio, sampleRate)
    # Comparison.plot_hist(originalAudio_hist, sampleRate, "Original Audio", 'b')
    ref_score = Comparison.hist_intersection(originalAudio_hist[:,0], originalAudio_hist[:,0])

    reducedOriginalAudio = np.round(originalAudio * SCALE)
    reducedOriginalAudio_hist = Comparison.freq_hist(reducedOriginalAudio, sampleRate)
    Comparison.plot_hist(reducedOriginalAudio_hist, sampleRate, "Reduced Original Audio", 'r')
    reducedAudio_score = Comparison.hist_intersection(reducedOriginalAudio_hist[:,0], originalAudio_hist[:,0])
    print("Percentage similar (Reduced Audio / Original Audio): {} %".format(round(100 * reducedAudio_score / ref_score, 4)))

    distortedAudio_hist = Comparison.freq_hist(distortedAudio, sampleRate)
    Comparison.plot_hist(distortedAudio_hist, sampleRate, "Distorted Audio", 'g')
    distortedAudio_score = Comparison.hist_intersection(distortedAudio_hist[:,0], originalAudio_hist[:,0])
    print("Percentage similar (Distorted Audio / Original Audio): {} %".format(round(100 * distortedAudio_score / ref_score, 4)))


    # Find regions of clipping
    clippedRegLeft = detectClipping(distortedAudio[:, 0])
    clippedRegRight = detectClipping(distortedAudio[:, 1])

    # Perform global amplitude reduction
    # Reduce by (1-SCALE)%
    reducedAudio = np.round(distortedAudio * SCALE)

    # Perform cubic spline interpolation for every clipped region
    cleanedAudio = np.zeros(reducedAudio.shape)
    cleanedAudio[:, 0] = interpolate(clippedRegLeft, reducedAudio[:, 0])
    cleanedAudio[:, 1] = interpolate(clippedRegRight, reducedAudio[:, 1])

    # FFT of the cleaned audio
    cleanedAudio_hist = Comparison.freq_hist(cleanedAudio, sampleRate)
    Comparison.plot_hist(cleanedAudio_hist, sampleRate, "Cleaned Audio", 'c')
    cleanedAudio_score = Comparison.hist_intersection(cleanedAudio_hist[:,0], originalAudio_hist[:,0])
    # We want the following ratio to be as close to 1.0 as possible:
    print("Percentage similar (Cleaned Audio / Original Audio): {} %".format(round(100 * cleanedAudio_score / ref_score, 4)))


    # Intelligent Scaling of Cleaned Audio
    reducedCleanAudio = Comparison.intelligent_reduction(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight)
    reducedCleanAudio = reducedCleanAudio.astype('int16')

    # Average max interpolation error
    avg_max_error = Comparison.avg_max_interp_error(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight)
    print("Average max interpolation error: {}".format(avg_max_error))

    # Produce a new wavfile. IMPORTANT: must be of type 'int16'
    wavfile.write('Trumpet_Cleaned_reduced.wav', sampleRate, reducedCleanAudio)

    #%% Plotting
    length = originalAudio.shape[0] / sampleRate
    time = np.linspace(0., length, originalAudio.shape[0])
    plt.figure(1)
    plt.title('Distorted Audio')
    # plt.plot(time, originalAudio[:, 0], label="Left channel")
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

    # Single Clipped Region -----------------------------------------------------------
    plot_single_region(1598, 1610, distortedAudio, originalAudio, sampleRate, 6)
    # ---------------------------------------------------------------------------------
