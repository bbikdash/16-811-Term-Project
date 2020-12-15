"""
@author: Bassam Bikdash, Ivan Cisneros
Testing.py

16-811 Term Project: Using Cubic Splines to Reconstruct Clipped Audio Signals
Due: December 2020
"""

import numpy as np
import AudioReconstruction


files = {'../test_audio/Trumpet_Original.wav':
         ['../test_audio/Trumpet_Distorted1.wav',
          '../test_audio/Trumpet_Distorted2.wav',
          '../test_audio/Trumpet_Distorted3.wav',
          '../test_audio/Trumpet_Distorted4.wav'],
         '../test_audio/Brahms_Original.wav':
         ['../test_audio/Brahms_Distorted1.wav',
          '../test_audio/Brahms_Distorted2.wav',
          '../test_audio/Brahms_Distorted3.wav',
          '../test_audio/Brahms_Distorted4.wav'],
         '../test_audio/Vibe_Original.wav':
         ['../test_audio/Vibe_Distorted1.wav',
          '../test_audio/Vibe_Distorted2.wav',
          '../test_audio/Vibe_Distorted3.wav',
          '../test_audio/Vibe_Distorted4.wav',
          '../test_audio/Vibe_Distorted5.wav']}



for o in files.keys():
    for d in files[o]:

        AudioReconstruction.reconstruct(o, d)





