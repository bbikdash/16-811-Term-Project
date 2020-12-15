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
THRESHOLD = 5900  # originally 3
CLIP = MAX_CLIP - THRESHOLD
SCALE = 0.85

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


def reconstructAndCompare(original_filename, distorted_filename):
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
    ref_score = Comparison.hist_intersection(originalAudio_hist[:,0], originalAudio_hist[:,0])

    reducedOriginalAudio = np.round(originalAudio * SCALE)
    reducedOriginalAudio_hist = Comparison.freq_hist(reducedOriginalAudio, sr)
    Comparison.plot_hist(reducedOriginalAudio_hist, sr, "Reduced Original Audio", 'r')
    reducedAudio_score = Comparison.hist_intersection(reducedOriginalAudio_hist[:,0], originalAudio_hist[:,0])
    print("Percentage similar (Reduced Audio / Original Audio): {} %".format(round(100 * reducedAudio_score / ref_score, 4)))

    distortedAudio_hist = Comparison.freq_hist(distortedAudio, sr)
    Comparison.plot_hist(distortedAudio_hist, sr, "Distorted Audio", 'g')
    distortedAudio_score = Comparison.hist_intersection(distortedAudio_hist[:,0], originalAudio_hist[:,0])
    print("Percentage similar (Distorted Audio / Original Audio): {} %".format(round(100 * distortedAudio_score / ref_score, 4)))

    # Find regions of clipping
    clippedRegLeft = detectClipping(distortedAudio[:, 0])
    clippedRegRight = detectClipping(distortedAudio[:, 1])

    perc_dist_left = Comparison.percentDistorted(clippedRegLeft, len(distortedAudio))
    perc_dist_right = Comparison.percentDistorted(clippedRegRight, len(distortedAudio))

    print('%.2f%% of the left audio channel is distorted.' % perc_dist_left)
    print('%.2f%% of the right audio channel is distorted.' % perc_dist_right)


    # Perform global amplitude reduction
    # Reduce by (1-SCALE)%
    reducedAudio = np.round(distortedAudio * SCALE)

    # Perform cubic spline interpolation for every clipped region
    cleanedAudio = np.zeros(reducedAudio.shape)
    cleanedAudio[:, 0] = interpolate(clippedRegLeft, reducedAudio[:, 0])
    cleanedAudio[:, 1] = interpolate(clippedRegRight, reducedAudio[:, 1])

    # FFT of the cleaned audio
    cleanedAudio_hist = Comparison.freq_hist(cleanedAudio, sr)
    Comparison.plot_hist(cleanedAudio_hist, sr, "Cleaned Audio", 'c')
    cleanedAudio_score = Comparison.hist_intersection(cleanedAudio_hist[:,0], originalAudio_hist[:,0])
    # We want the following ratio to be as close to 1.0 as possible:
    print("Percentage similar (Cleaned Audio / Original Audio): {} %".format(round(100 * cleanedAudio_score / ref_score, 4)))

    # Intelligent Scaling of Cleaned Audio
    reducedCleanAudio = Comparison.intelligent_reduction(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight)
    reducedCleanAudio = reducedCleanAudio.astype('int16')   # Use 16 bit signed int when writing to wav file

    # Average max interpolation error
    avg_max_error = Comparison.avg_max_interp_error(cleanedAudio, originalAudio, clippedRegLeft, clippedRegRight)
    # print("Average max interpolation error: {}".format(avg_max_error))
    print("Average max interpolation error: {} dB".format(avg_max_error))

    # Single Clipped Region for Trumpet Audio -----------------------------------------
    # Plotting.plot_single_region(1598, 1610, distortedAudio, originalAudio, reducedAudio, cleanedAudio, sampleRate, 6)
    # ---------------------------------------------------------------------------------

    Plotting.generalPlot(originalAudio, distortedAudio, sr)
    Plotting.createSubplot(originalAudio, reducedAudio, reducedCleanAudio, sr)

    return cleanedAudio, sr

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

    clean, sr = reconstructAndCompare('../test_audio/Vibe_Original.wav',
                                      '../test_audio/Vibe_Distorted1.wav')

    # clean, sr = reconstruct('../test_audio/Chopin - Sonata No. 2 in B flat minor, Op. 35 [Pogorelich].wav')
    # exportAudio('../test_audio/Chopin Clean.wav', sr, clean)
