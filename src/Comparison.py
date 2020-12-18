"""
@author: Bassam Bikdash, Ivan Cisneros
Comparison.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

class Comparison:
    @staticmethod
    def freq_hist(audio, sampleRate):
        """
        Returns the normalized Fast Fourier Transform for a single audio file.
        Assumes the audio input is two channel. Returns a FFT for each channel
        in the same two channel format as the input audio matrix.
        Normalizes the FFT by dividing all values by the peak value in the
        frequency space (for that particular audio channel).
        :param audio: The audio waveform matrix of size (n,2)
        :param sampleRate: The sample rate returned by scipy.io.wavfile.read()
        :return hist: A normalized (peak value is 1) two channel histogram matrix of
            size (m,2). hist[:,0] corresponds to audio[:,0] and
            hist[:,1] corresponds to audio[:,1].
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
        """
        Plots the two channels of the input histogram hist.
        :param hist: The normalized input histogram of size (m,2)
        :param sampleRate: The sample rate returned by scipy.io.wavfile.read()
        :param title: The title of the plot. A string.
        :param color: The color to plot. Usually a char. Check matplotlib documentation.
        :return: None.
        """
        n = len(hist)
        T = 1/sampleRate
        xf = np.linspace(0.0, 1.0/(2.0*T), n)

        hist_left = hist[:,0]
        hist_right = hist[:,1]

        fig, axs = plt.subplots(2)
        fig.suptitle(title)
        axs[0].plot(xf, hist_left, color=color)
        axs[0].grid()
        # axs[0].setTitle('')

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
        :param hist1: A single channel of a histogram. size (m,1).
        :param hist2: A single channel of a histogram. size (m,1).
        :return sim: The similarity score.
        """
        sim = np.sum(np.minimum(hist1, hist2))
        # print(sim)
        return sim


    @staticmethod
    def intelligent_reduction(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight):
        """
        Using clippedRegLeft and/or clippedRegRight, finds the biggest region of the waveform
        cleanedAudio that had no clipped values, stores this as a range, then finds the
        peak value of this range in both cleanedAudio and originalAudio in order to form a scaling
        ratio such that when cleanedAudio is multiplied by this ratio it will be of approximately the
        same magnitude as originalAudio.
        Currently only does this for the left channel.
        :param cleanedAudio:  The reconstructed audio matrix  of size (n,2).
        :param originalAudio: The original audio that corresponds to the cleanedAudio. Size (n,2).
        :param clippedRegLeft: The list of clipped regions of the left channel of cleanedAudio.
        :param clippedRegRight: The list of clipped regions of the right channel of cleanedAudio.
        :return reduced_cleanedAudio: The scaled version of cleanedAudio.
        """
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
        reduced_cleanedAudio = np.round(cleanedAudio * scale)

        return reduced_cleanedAudio


    @staticmethod
    def avg_max_interp_error(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight):
        """
        Calculates the average maximum interpolation error over the entire audio waveform. Compares each region
        timestamped in clippedRegLeft to the same timestamp in originalAudio by first scaling the cleanedAudio portion
        to be the same maximum magnitude as originalAudio, and then takes the absolute difference. Averages the
        max interpolation errors over the number of clipped regions (length of clippedRegLeft and/or clippedRegRight).
        Currently only does this for the left channel.
        :param cleanedAudio:  The reconstructed audio matrix  of size (n,2).
        :param originalAudio: The original audio that corresponds to the cleanedAudio. Size (n,2).
        :param clippedRegLeft: The list of clipped regions of the left channel of cleanedAudio.
        :param clippedRegRight: The list of clipped regions of the right channel of cleanedAudio.
        :return avg_max_error: The average maximum interpolation error.
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

        avg_max_error = avg_max_error / N
        # Convert avg max error to decibel
        return avg_max_error

    @staticmethod
    def percentDistorted(regions, sizeOfAudio):
        """
        Compute the percent of the audio that is distorted
        """
        total = 0
        for r in regions:
            total += r[1] - r[0]
        return (total / sizeOfAudio) * 100.0


