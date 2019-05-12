# -*- coding: utf-8 -*-
"""

@author: Kyle Watters / Sumit Mondal
# Note that this code will not work without the required data set ! If you have interest in that data set please contact me at sumitmondal@gatech.edu.

# A little about the data set:
# There is approximately 100-300 whistles for each species, of which there are about 12 species.
# Most of the whistles were gathered from the WHO(Woods Hole Oceanographic) Institution.

"""

import numpy as np
from numpy import newaxis
import random
import matplotlib.pyplot as plt
from sklearn import neighbors
import os
import scipy.io.wavfile
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
global num_fcc_frames
global num_ceps
global min_length_secs

num_fcc_frames = 50
num_ceps = 12
min_length_secs = .5

# get N+2 values linearly across the mel scale
def get_mel_points (max_rate, min_rate, N):
    low_freq_mel = (2595 * np.log10(1 + (min_rate / 2) / 700))  # Convert Hz to Mel
    high_freq_mel = (2595 * np.log10(1 + (max_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, N + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    return hz_points;

# get N+2 values linearly across the logistic scale
def get_log_points (max_rate, min_rate, N):
    low_freq_log = 41000/(1+np.exp(-0.0002*(min_rate-20000)))
    high_freq_log = 41000/(1+np.exp(-0.0002*(max_rate-20000)))  # Convert Hz to Logistic
    log_points = np.linspace(low_freq_log, high_freq_log, N + 2)  # Equally spaced in Logistic scale
    hz_points = -5000*np.log(45000/log_points - 1)+20000  # Convert Logistic to Hz
    return hz_points;

# plot FCC
def plot_FCC (fcc_data, plot_title):
    fig = plt.figure(figsize=(15, 2))
    
    ax = fig.add_subplot(111)
    ax.set_title(plot_title)
    plt.imshow(fcc_data.T)
    ax.set_aspect('equal')
    
    cax = fig.add_axes([0.25, 0.1, 0.9, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
    return;

# get feature vector from signal
# get feature vector from signal
def get_FCC(signal, sample_rate, hz_points):
    
    # pre emphasis on the signal
    pre_emphasis = 0.97
        
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # frame size and overlap in seconds
    frame_size = 0.025
    frame_stride = 0.01
    
    # frame step in samples based on sampling rate
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) 
    
    # create frames from the original signal
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # use a hamming window on each frame 
    frames *= np.hamming(frame_length)
    
    # size of the FFT 
    NFFT = 512
    
    # run the fft and calculate the power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    # choose values of the power spectrum based on the mel or log scaled points
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    
    nfilt = 40
    
    # ifft and log functon on the power spectrum
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    # run discrte cosine transform on each frame
    log_fcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-(ceps+1)
    
    # shpae into an array
    (nframes, ncoeff) = log_fcc.shape
    
    # set to zero mean
    log_fcc -= (np.mean(log_fcc, axis=0) + 1e-8)

    # extract the num_log
    s = int(np.size(log_fcc)/num_ceps)
    
    if s > num_fcc_frames:
        var_lfcc = np.var(log_fcc, axis=1)
        start = int(np.argmax(var_lfcc) - num_fcc_frames/2)
        if start < 0:
            start = 0
        if (start + num_fcc_frames) > s:
            start = s - num_fcc_frames
        log_fcc = log_fcc[start:num_fcc_frames+start, :]
    elif s < num_fcc_frames:
        extra = np.zeros((num_fcc_frames-s, num_ceps), dtype = 'float64')
        log_fcc = np.concatenate((log_fcc, extra), axis = 0) 
    
    return log_fcc;

# get first [num_fcc_frames] Logistic FCC vectors and place them in arrays by species
path = os.getcwd()

if True: # create new feature sets if parameters have changed
    
    mel_points = get_mel_points(40000, 3000, 40)
    log_points = get_log_points(40000, 3000, 40)
            
    #Melon Headed Whale (Peponocephala electra)
    dirs = os.listdir( path + '/Melon Headed Whale (Peponocephala electra)' )
    
    Melon_Headed_Whale = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Melon_Headed_Whale_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Melon_Headed_Whale_tensorflow = np.empty((0,num_fcc_frames,num_ceps), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Melon Headed Whale (Peponocephala electra)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Melon_Headed_Whale = np.append(Melon_Headed_Whale, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Melon_Headed_Whale_Mel = np.append(Melon_Headed_Whale_Mel, vector_MFCC, axis = 0)
            #count = count + 1
            #plot_LFCC(LFCC, ['Melon Headed Whale (Peponocephala electra) ' + str(count)])
    
    
    #Bottlenose Dolphin (Tursiops truncatus)
    dirs = os.listdir( path + '/Bottlenose Dolphin (Tursiops truncatus)' )
    
    Bottlenose_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Bottlenose_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Bottlenose Dolphin (Tursiops truncatus)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Bottlenose_Dolphin = np.append(Bottlenose_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Bottlenose_Dolphin_Mel = np.append(Bottlenose_Dolphin_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Bottlenose Dolphin (Tursiops truncatus) ' + str(count)])
           
            
    #Clymene Dolphin (Stenella clymene)
    dirs = os.listdir( path + '/Clymene Dolphin (Stenella clymene)' )
    
    Clymene_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Clymene_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Clymene Dolphin (Stenella clymene)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Clymene_Dolphin = np.append(Clymene_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Clymene_Dolphin_Mel = np.append(Clymene_Dolphin_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Clymene Dolphin (Stenella clymene) ' + str(count)])
    
    
    #Common Dolphin (Delphinus delphis)
    dirs = os.listdir( path + '/Common Dolphin (Delphinus delphis)' )
    
    Common_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Common_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Common Dolphin (Delphinus delphis)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Common_Dolphin = np.append(Common_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Common_Dolphin_Mel = np.append(Common_Dolphin_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Common Dolphin (Delphinus delphis) ' + str(count)])
    
    
    #False Killer Whale (Pseudorca crassidens)
    dirs = os.listdir( path + '/False Killer Whale (Pseudorca crassidens)' )
    
    False_Killer_Whale = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    False_Killer_Whale_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'False Killer Whale (Pseudorca crassidens)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                False_Killer_Whale = np.append(False_Killer_Whale, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                False_Killer_Whale_Mel = np.append(False_Killer_Whale_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['False Killer Whale (Pseudorca crassidens) ' + str(count)])
    
    
    #Panatropical Spotted Dolphin (Stenella attenuata)
    dirs = os.listdir( path + '/Panatropical Spotted Dolphin (Stenella attenuata)' )
    
    Panatropical_Spotted_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Panatropical_Spotted_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Panatropical Spotted Dolphin (Stenella attenuata)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Panatropical_Spotted_Dolphin = np.append(Panatropical_Spotted_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Panatropical_Spotted_Dolphin_Mel = np.append(Panatropical_Spotted_Dolphin_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Panatropical Spotted Dolphin (Stenella attenuata) ' + str(count)])
    
    
    #Rissos Dolphin (Grampus griseus)
    dirs = os.listdir( path + '/Rissos Dolphin (Grampus griseus)' )
    
    Rissos_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Rissos_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Rissos Dolphin (Grampus griseus)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Rissos_Dolphin = np.append(Rissos_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Rissos_Dolphin_Mel = np.append(Rissos_Dolphin_Mel, vector_MFCC, axis = 0)
            #count = count + 1
            #plot_LFCC(LFCC, ['Rissos Dolphin (Grampus griseus) ' + str(count)])
    
    
    #Short-Finned Pilot Whale (Globicephala macrorhynchus)
    dirs = os.listdir( path + '/Short-Finned Pilot Whale (Globicephala macrorhynchus)' )
    
    Short_Finned_Pilot_Whale = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Short_Finned_Pilot_Whale_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Short-Finned Pilot Whale (Globicephala macrorhynchus)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Short_Finned_Pilot_Whale = np.append(Short_Finned_Pilot_Whale, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Short_Finned_Pilot_Whale_Mel = np.append(Short_Finned_Pilot_Whale_Mel, vector_MFCC, axis = 0)
            #count = count + 1
            #plot_LFCC(LFCC, ['Short-Finned Pilot Whale (Globicephala macrorhynchus) ' + str(count)])
    
    
    #Sperm Whale (Physeter macrocephalus)
    dirs = os.listdir( path + '/Sperm Whale (Physeter macrocephalus)' )
    
    Sperm_Whale = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Sperm_Whale_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Sperm Whale (Physeter macrocephalus)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Sperm_Whale = np.append(Sperm_Whale, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Sperm_Whale_Mel = np.append(Sperm_Whale_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Sperm Whale (Physeter macrocephalus) ' + str(count)])
    
    
    #Spinner Dolphin (Stenella longirostris)
    dirs = os.listdir( path + '/Spinner Dolphin (Stenella longirostris)' )
    
    Spinner_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Spinner_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Spinner Dolphin (Stenella longirostris)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Spinner_Dolphin = np.append(Spinner_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Spinner_Dolphin_Mel = np.append(Spinner_Dolphin_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Spinner Dolphin (Stenella longirostris) ' + str(count)])
    
    
    #Striped Dolphin (Stenella coeruleoalba)
    dirs = os.listdir( path + '/Striped Dolphin (Stenella coeruleoalba)' )
    
    Striped_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    Striped_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'Striped Dolphin (Stenella coeruleoalba)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                Striped_Dolphin = np.append(Striped_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                Striped_Dolphin_Mel = np.append(Striped_Dolphin_Mel, vector_MFCC, axis = 0)
                
            #count = count + 1
            #plot_LFCC(LFCC, ['Striped Dolphin (Stenella coeruleoalba) ' + str(count)])
    
    
    #White-sided Dolphin (Lagenorhynchus acutus)
    dirs = os.listdir( path + '/White-sided Dolphin (Lagenorhynchus acutus)' )
    
    White_sided_Dolphin = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    White_sided_Dolphin_Mel = np.empty((0,num_ceps*num_fcc_frames), dtype='float64')
    count = 0
    for file in dirs:
        if (file.endswith(".wav")):
            LFCC = []
            full_name = 'White-sided Dolphin (Lagenorhynchus acutus)/' + file
            samp_rate, sig = scipy.io.wavfile.read(full_name)  # File assumed to be in the same directory
            sig = sig[0:int(3 * samp_rate)]  # Keep the first 3 seconds
            if (samp_rate >= 81920) & (np.size(sig)/samp_rate > min_length_secs) & (count < 3000):
                count += 1
                # print(file)
                LFCC = get_FCC(sig, samp_rate, log_points)
                vector_Log_FCC = np.reshape(LFCC, (-1, num_ceps*num_fcc_frames))
                White_sided_Dolphin = np.append(White_sided_Dolphin, vector_Log_FCC, axis = 0)
                
                MFCC = get_FCC(sig, samp_rate, mel_points)
                vector_MFCC = np.reshape(MFCC, (-1, num_ceps*num_fcc_frames))
                White_sided_Dolphin_Mel = np.append(White_sided_Dolphin_Mel, vector_MFCC, axis = 0)

            #count = count + 1
            #plot_LFCC(LFCC, ['White-sided Dolphin (Lagenorhynchus acutus) ' + str(count)])
    
    # Create full array of other species
    
    np.save( path + '/NumpyArrays/Clymene_Dolphin', Clymene_Dolphin)
    
    np.save(path + '/NumpyArrays/Common_Dolphin', Common_Dolphin)
    
    np.save(path + '/NumpyArrays/False_Killer_Whale', False_Killer_Whale)
    
    np.save(path + '/NumpyArrays/Melon_Headed_Whale', Melon_Headed_Whale)
    
    np.save(path + '/NumpyArrays/Panatropical_Spotted_Dolphin', Panatropical_Spotted_Dolphin)
    
    np.save(path + '/NumpyArrays/Rissos_Dolphin', Rissos_Dolphin)
    
    np.save(path + '/NumpyArrays/Short_Finned_Pilot_Whale', Short_Finned_Pilot_Whale)
    
    np.save(path + '/NumpyArrays/Sperm_Whale', Sperm_Whale)
    
    np.save(path + '/NumpyArrays/Spinner_Dolphin', Spinner_Dolphin)
    
    np.save(path + '/NumpyArrays/Striped_Dolphin', Striped_Dolphin)
    
    np.save(path + '/NumpyArrays/White_sided_Dolphin', White_sided_Dolphin)
    
    np.save(path + '/NumpyArrays/Bottlenose_Dolphin', Bottlenose_Dolphin)
    
    # Mel Frequency
    
    np.save(path + '/NumpyArrays/Clymene_Dolphin_Mel', Clymene_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/Common_Dolphin_Mel', Common_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/False_Killer_Whale_Mel', False_Killer_Whale_Mel)
    
    np.save(path + '/NumpyArrays/Melon_Headed_Whale_Mel', Melon_Headed_Whale_Mel)
    
    np.save(path + '/NumpyArrays/Panatropical_Spotted_Dolphin_Mel', Panatropical_Spotted_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/Rissos_Dolphin_Mel', Rissos_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/Short_Finned_Pilot_Whale_Mel', Short_Finned_Pilot_Whale_Mel)
    
    np.save(path + '/NumpyArrays/Sperm_Whale_Mel', Sperm_Whale_Mel)
    
    np.save(path + '/NumpyArrays/Spinner_Dolphin_Mel', Spinner_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/Striped_Dolphin_Mel', Striped_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/White_sided_Dolphin_Mel', White_sided_Dolphin_Mel)
    
    np.save(path + '/NumpyArrays/Bottlenose_Dolphin_Mel', Bottlenose_Dolphin_Mel)

    
    
