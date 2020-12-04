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

