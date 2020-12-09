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

        for i in range(indices[0], indices[1]):
            data[i] = cs(i)

    return data


if __name__ == "__main__":
    """
    For all two channel matrices, we will adopt the convention:
        - left channel: audio[:,0]
        - right channel: audio[:,1]
    """
    # Essential ------------------------------------------------------------------------------------------
    # Load the audio as a waveform `y`
    # Store the sampling rate as `sr
    filename = '../test_audio/Trumpet_Distorted1.wav'
    sampleRate, distortedAudio = wavfile.read(filename)
    distortedAudio = distortedAudio.astype('int32')
    # print("distortedAudio.shape: {}".format(distortedAudio.shape))

    filename = '../test_audio/Trumpet_Original.wav'
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
    # wavfile.write('Trumpet_Cleaned_reduced.wav', sampleRate, reducedCleanAudio)

    