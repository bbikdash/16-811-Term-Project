"""
@author: Bassam Bikdash, Ivan Cisneros
AudioReconstruction.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from Interpolation import Interpolation
from Comparison import Comparison
from Plotting import Plotting

# These represent the bounds
MAX_CLIP = 32767
MIN_CLIP = -32768
THRESHOLD = 10  # originally 5900
SCALE = 0.85

def detectClipping(data, thresh):
    clipped_region = []  # List containing clipped regions (start and end indices)

    inMiddle = False
    bounds = np.zeros(2, dtype=int)
    clip = MAX_CLIP - thresh

    for i in range(len(data)):

        if abs(data[i]) >= clip and inMiddle == False:
            # ASSERT: Found the start of a clipped region
            bounds[0] = i - 1  # Off by one error found here
            inMiddle = True
        elif abs(data[i]) < clip and inMiddle == True:
            # ASSERT: Found the end of a clipped region
            bounds[1] = i
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
        # cs = Interpolation.cubic_spline(indices, data, numpoints)

        for i in range(indices[0], indices[1]):
            data[i] = cs(i)

    return data


def reconstructAndCompare(original_filename, distorted_filename, plot=True):
    """
    For all two channel matrices, we will adopt the convention:
        - left channel: audio[:,0]
        - right channel: audio[:,1]
    """
    # Essential ------------------------------------------------------------------------------------------
    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr
    sr, originalAudio = wavfile.read(original_filename)
    originalAudio = originalAudio.astype('int32')

    sr, distortedAudio = wavfile.read(distorted_filename)
    distortedAudio = distortedAudio.astype('int32')


    # Comparing Cleaning Quality through FFT -----------------------------------------------------------
    # FFT histograms
    originalAudio_hist = Comparison.freq_hist(originalAudio, sr)
    left_ref_score = Comparison.hist_intersection(originalAudio_hist[:,0], originalAudio_hist[:,0])
    right_ref_score = Comparison.hist_intersection(originalAudio_hist[:,1], originalAudio_hist[:,1])

    reducedOriginalAudio = np.round(originalAudio * SCALE)
    reducedOriginalAudio_hist = Comparison.freq_hist(reducedOriginalAudio, sr)
    if plot:
        Comparison.plot_hist(reducedOriginalAudio_hist, sr, "Reduced Original Audio", 'r')
    left_reduced_Audio_score = Comparison.hist_intersection(reducedOriginalAudio_hist[:,0], originalAudio_hist[:,0])
    right_reduced_Audio_score = Comparison.hist_intersection(reducedOriginalAudio_hist[:,1], originalAudio_hist[:,1])

    print("Percentage similar (Left Reduced Audio / Left Original Audio): {} %".format(round(100 * left_reduced_Audio_score / left_ref_score, 4)))
    print("Percentage similar (Right Reduced Audio / Left Original Audio): {} %".format(round(100 * right_reduced_Audio_score / right_ref_score, 4)))

    distortedAudio_hist = Comparison.freq_hist(distortedAudio, sr)
    if plot:
        Comparison.plot_hist(distortedAudio_hist, sr, "Distorted Audio", 'g')
    left_distorted_Audio_score = Comparison.hist_intersection(distortedAudio_hist[:,0], originalAudio_hist[:,0])
    right_distorted_Audio_score = Comparison.hist_intersection(distortedAudio_hist[:,1], originalAudio_hist[:,1])

    print("Percentage similar (Left Distorted Audio / Left Original Audio): {} %".format(round(100 * left_distorted_Audio_score / left_ref_score, 4)))
    print("Percentage similar (Right Distorted Audio / Right Original Audio): {} %".format(round(100 * right_distorted_Audio_score / right_ref_score, 4)))

    # Perform global amplitude reduction
    # Reduce by (1-SCALE)%
    reducedAudio = np.round(distortedAudio * SCALE)

    # Perform cubic spline interpolation for every clipped region
    cleanedAudio = np.zeros(reducedAudio.shape)

    # thresh_levels = [5, 10, 50, 250, 500]
    # thresh_levels = [2500, 5000, 8000]
    # best_score = 0
    # best_thresh = 0
    # for thresh in thresh_levels:
    #     # Find regions of clipping
    #     clippedRegLeft = detectClipping(distortedAudio[:, 0], thresh)
    #     clippedRegRight = detectClipping(distortedAudio[:, 1], thresh)
    #
    #     # perc_dist_left = Comparison.percentDistorted(clippedRegLeft, len(distortedAudio))
    #     # perc_dist_right = Comparison.percentDistorted(clippedRegRight, len(distortedAudio))
    #     #
    #     # print('%.2f%% of the left audio channel is distorted.' % perc_dist_left)
    #     # print('%.2f%% of the right audio channel is distorted.' % perc_dist_right)
    #
    #     cleanedAudio[:, 0] = interpolate(clippedRegLeft, reducedAudio[:, 0])
    #     cleanedAudio[:, 1] = interpolate(clippedRegRight, reducedAudio[:, 1])
    #
    #     # FFT of the cleaned audio
    #     cleanedAudio_hist = Comparison.freq_hist(cleanedAudio, sr)
    #     # Comparison.plot_hist(cleanedAudio_hist, sr, "Cleaned Audio", 'c')
    #     cleanedAudio_score = Comparison.hist_intersection(cleanedAudio_hist[:,0], originalAudio_hist[:,0])
    #     # We want the following ratio to be as close to 1.0 as possible:
    #     curr_score = round(100 * cleanedAudio_score / ref_score, 4)
    #     print("Current Percentage similar (Cleaned Audio / Original Audio): {} %".format(curr_score))
    #
    #     if curr_score > best_score:
    #         print("New best score found. best_score: {},   curr_score: {}".format(best_score, curr_score))
    #         best_score = curr_score
    #         best_thresh = thresh
    #
    # print("Best Percentage similar (Cleaned Audio / Original Audio): {} %".format(best_score))
    # print("Best threshold: {}".format(best_thresh))

    thresh = THRESHOLD

    # Find regions of clipping
    clippedRegLeft = detectClipping(distortedAudio[:, 0], thresh)
    clippedRegRight = detectClipping(distortedAudio[:, 1], thresh)

    perc_dist_left = Comparison.percentDistorted(clippedRegLeft, len(distortedAudio))
    perc_dist_right = Comparison.percentDistorted(clippedRegRight, len(distortedAudio))
    print('%.2f%% of the left audio channel is distorted.' % perc_dist_left)
    print('%.2f%% of the right audio channel is distorted.' % perc_dist_right)

    cleanedAudio[:, 0] = interpolate(clippedRegLeft, reducedAudio[:, 0])
    cleanedAudio[:, 1] = interpolate(clippedRegRight, reducedAudio[:, 1])

    # FFT of the cleaned audio
    cleanedAudio_hist = Comparison.freq_hist(cleanedAudio, sr)
    if plot:
        Comparison.plot_hist(cleanedAudio_hist, sr, "Cleaned Audio", 'c')
    left_cleaned_Audio_score = Comparison.hist_intersection(cleanedAudio_hist[:, 0], originalAudio_hist[:, 0])
    right_cleaned_Audio_score = Comparison.hist_intersection(cleanedAudio_hist[:, 1], originalAudio_hist[:, 1])

    # We want the following ratio to be as close to 1.0 as possible:
    left_curr_score = round(100 * left_cleaned_Audio_score / left_ref_score, 4)
    right_curr_score = round(100 * right_cleaned_Audio_score / right_ref_score, 4)

    print("Percentage similar (Left Cleaned Audio / Left Original Audio): {} %".format(left_curr_score))
    print("Percentage similar (Right Cleaned Audio / Right Original Audio): {} %".format(right_curr_score))

    # Intelligent Scaling of Cleaned Audio
    reducedCleanAudio = Comparison.intelligent_reduction(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight)
    reducedCleanAudio = reducedCleanAudio.astype('int16')   # Use 16 bit signed int when writing to wav file

    # Average max interpolation error
    avg_max_error = Comparison.avg_max_interp_error(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight)
    print("Average max interpolation error (raw): {}".format(avg_max_error))
    print("Average max interpolation error (dB): {} dB".format(np.log10(avg_max_error)))

    # Single Clipped Region for Trumpet Audio -----------------------------------------
    # Plotting.plot_single_region(1598, 1610, distortedAudio, originalAudio, reducedAudio, cleanedAudio, sampleRate, 6)
    # ---------------------------------------------------------------------------------

    if plot:
        Plotting.generalPlot(originalAudio, distortedAudio, sr)
        Plotting.createSubplot(originalAudio, reducedAudio, reducedCleanAudio, sr)

    return reducedCleanAudio, sr

def reconstruct(distorted_filename):
    """
    For all two channel matrices, we will adopt the convention:
        - left channel: audio[:,0]
        - right channel: audio[:,1]
    """
    # Essential ------------------------------------------------------------------------------------------
    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr
    sr, distortedAudio = wavfile.read(distorted_filename)

    # Find regions of clipping
    clippedRegLeft = detectClipping(distortedAudio[:, 0])
    clippedRegRight = detectClipping(distortedAudio[:, 1])

    perc_dist_left = Comparison.percentDistorted(clippedRegLeft, len(distortedAudio))
    perc_dist_right = Comparison.percentDistorted(clippedRegRight, len(distortedAudio))

    print('%.2f%% of the left audio channel is distorted.' % perc_dist_left)
    print('%.2f%% of the right audio channel is distorted.' % perc_dist_right)

    # Perform global amplitude reduction
    reducedAudio = np.round(distortedAudio * 0.75)

    # Perform cubic spline interpolation for every clipped region
    cleanedAudio = np.zeros(reducedAudio.shape)
    cleanedAudio[:, 0] = interpolate(clippedRegLeft, reducedAudio[:, 0])
    cleanedAudio[:, 1] = interpolate(clippedRegRight, reducedAudio[:, 1])

    cleanedAudio = cleanedAudio.astype('int16')   # Use 16 bit signed int when writing to wav file

    return cleanedAudio, sr


def exportAudio(name, sampleRate, cleanedAudio):
    # Produce a new wavfile. IMPORTANT: must be of type 'int16'
    wavfile.write(name, sampleRate, cleanedAudio)



if __name__ == "__main__":

    clean, sr = reconstructAndCompare('../test_audio/Trumpet_Original.wav',
                                      '../test_audio/Trumpet_Distorted1.wav')

    # clean, sr = reconstruct('../test_audio/Chopin - Sonata No. 2 in B flat minor, Op. 35 [Pogorelich].wav')
    # exportAudio('../test_audio/Chopin Clean.wav', sr, clean)
