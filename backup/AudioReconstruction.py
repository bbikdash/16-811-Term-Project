"""
@author: Bassam Bikdash, Ivan Cisneros
Neural Networks

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import CubicSpline
import scipy

# These represent the bounds
MAX_CLIP = 32767
MIN_CLIP = -32768
THRESHOLD = 5900  # originally 3
CLIP = MAX_CLIP - THRESHOLD
# SCALE is arbitrary. Can also experiment with this to see if affects
# performance.
SCALE = 0.85


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




class Comparison:
    @staticmethod
    def freq_hist(audio, sampleRate):
        """
        Returns the normalized Fast Fourier Transform for a single audio file.
        Assumes the audio input is two channel. Returns a FFT for each channel
        in the same two channel format as the input audio matrix.
        Normalizes the FFT by dividing all values by the peak value in the
        frequency space (for that particular audio channel).
        """
        n = len(audio)
        T = 1/sampleRate
        yf_left = scipy.fft.fft(audio[:,0])
        yf_left = 2.0/n * np.abs(yf_left[:n//2])
        yf_right = scipy.fft.fft(audio[:,1])
        yf_right = 2.0/n * np.abs(yf_right[:n//2])

        # Normalize by dividing by the peak value:
        yf_left /= yf_left.max()
        yf_right /= yf_right.max()

        hist = np.stack((yf_left, yf_right), axis=1)
        return hist

    @staticmethod
    def plot_hist(hist, sampleRate, title, color):
        n = len(hist)
        T = 1/sampleRate
        xf = np.linspace(0.0, 1.0/(2.0*T), n)

        hist_left = hist[:,0]
        hist_right = hist[:,1]

        fig, axs = plt.subplots(2)
        fig.suptitle(title)
        axs[0].plot(xf, hist_left, color=color)
        axs[0].grid()
        axs[1].plot(xf, hist_right, color=color)
        axs[1].grid()
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.show()

    @staticmethod
    def hist_intersection(hist1, hist2):
        """
        Given two histograms, gives the intersection score. Higher is better.
        This type of comparison is volume invariant. Meaning, our scale
        variable doesn't affect the histogram comparison.
        """
        sim = np.sum(np.minimum(hist1, hist2))
        # print(sim)
        return sim

    @staticmethod
    def intelligent_reduction(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight):
        maxbins, _ = cleanedAudio.shape
        good_region_size = 10000
        regionbound_start = None
        regionbound_end = None
        num_regions = len(clippedRegLeft)
        best_region_size = 0
        for i in range(1, num_regions):
            start_im1 = clippedRegLeft[i-1][0]
            end_im1 = clippedRegLeft[i-1][1]

            start_i = clippedRegLeft[i][0]
            end_i = clippedRegLeft[i][1]

            if (start_i - end_im1) > good_region_size:
                regionbound_start = end_im1 + 1
                regionbound_end = start_i - 1
                # best_region_size = start_i - end_im1
            if (i == (num_regions-1)) and (regionbound_start is None):
                regionbound_start = end_i + 1
                regionbound_end = maxbins

        # print("regionbound_start: {}".format(regionbound_start))
        # print("regionbound_end: {}".format(regionbound_end))
        originalAudio_portion = np.abs(originalAudio[regionbound_start:regionbound_end,0]).max()
        cleanedAudio_portion = np.abs(cleanedAudio[regionbound_start:regionbound_end,0]).max()
        scale = originalAudio_portion/cleanedAudio_portion

        # print(scale)

        return np.round(cleanedAudio * scale)


    @staticmethod
    def avg_max_interp_error(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight):
        """
        Calculates the average maximum interpolation error over the entire audio waveform. Compares each region
        timestamped in clippedRegLeft to the same timestamp in originalAudio by first scaling the cleanedAudio portion
        to be the same maximum magnitude as originalAudio, and then takes the absolute difference.
        """
        N = len(clippedRegLeft)
        avg_max_error = 0
        i = 0
        for region in clippedRegLeft:
            start = region[0]
            end = region[1]
            originalAudio_portion = originalAudio[start:end, 0]
            cleanedAudio_portion = cleanedAudio[start:end, 0]
            originalAudio_max = np.abs(originalAudio_portion).max()
            cleanedAudio_max = np.abs(cleanedAudio_portion).max()
            scale = originalAudio_max / cleanedAudio_max
            cleanedAudio_portion = cleanedAudio_portion * scale

            avg_max_error += np.amax(np.abs(originalAudio_portion - cleanedAudio_portion))
            # if i % 2 == 0:
            #     num_points = len(originalAudio_portion)
            #     time = np.linspace(0,1,num_points)
            #
            #     plt.plot(time, originalAudio_portion, label="Original Audio Left Ch")
            #     plt.plot(time, cleanedAudio_portion, label="Cleaned Audio Left Ch")
            #     plt.legend()
            #     plt.xlabel("Time [s]")
            #     plt.ylabel("Amplitude")
            #     plt.show()
            #
            # i += 1

        return avg_max_error / N








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

        # if np.all(indices):
        #     continue

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
