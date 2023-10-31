import os
import h5py
import numpy as np
import pandas as pd
import scipy
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
import antropy as an 
from scipy.stats import skew, kurtosis
import seaborn as sns
from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
from scipy.signal import butter, lfilter
folder_path = '/Users/siddharth/Downloads/EEG Data Associated with Mental Workload/'
output_folder = '/Users/siddharth/Downloads/EEG Data Plots/'
os.makedirs(output_folder, exist_ok=True)
rms_final=[]
spectral_entropy_values = [] 
perm_entropy_value=[]
fd_values = []
sample_entropy_values=[]
svd_entropy_value=[]
detrended_fluctuation_values = []
Kurtosis_values = []
mean_values=[]
mmax_values=[]
mean_psd=[]
std_psd=[]
Approximate_E=[]
var_values=[]
Labels=[]
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

lowcut = 0.5  # Lower cutoff frequency in Hz
highcut = 4.0  # Upper cutoff frequency in Hz
fs = 1000  # Sampling frequency in Hz
order = 6  # Filter order

for filename in os.listdir(folder_path):
    if filename.endswith('.mat'):
        mat_file = os.path.join(folder_path, filename)
        with h5py.File(mat_file, 'r') as mat_data:
            eeg_data = np.transpose(mat_data['EEG'][:])
        # print(eeg_data.shape)
        for i in range(eeg_data.shape[0]):
            channel = butter_bandpass_filter(eeg_data[i, :], lowcut, highcut, fs, order)
            signal=eeg_data[i, :]

        fft_results = scipy.fft.fft(eeg_data)
        magnitude_spectrum = np.abs(fft_results)
        for i in range(eeg_data.shape[0]):
            if 'hi' in filename:
                Labels.append(1) 
            if 'lo' in filename:
                Labels.append(0)
            channel = eeg_data[i, :]
            signal=eeg_data[i, :]
            f, Pxx =scipy.signal.welch(channel,fs)
            mn=mean(Pxx)
            mean_psd.append(mn)
            sd=std(Pxx)
            std_psd.append(sd)
            # rms=np.sqrt(np.mean(signal**2))
            # rms_final.append(rms)
            spectral_entropy = np.mean(an.spectral_entropy(channel, 100, method='welch', normalize=True))
            spectral_entropy_values.append(spectral_entropy)
            perm_entropy = an.perm_entropy(channel, order=3, normalize=True)
            perm_entropy_value.append(perm_entropy)
            fd=an.katz_fd(channel)
            fd_values.append(fd)
            detrended_fluctuation=an.detrended_fluctuation(channel)
            detrended_fluctuation_values.append(detrended_fluctuation)
            sample_entropy=an.sample_entropy(channel, order=2, metric='chebyshev')
            sample_entropy_values.append(sample_entropy)
            svd_entropy=an.svd_entropy(channel, order=3, delay=1, normalize=True)
            svd_entropy_value.append(svd_entropy)
            Kurtosis=kurtosis(channel)
            Kurtosis_values.append(Kurtosis)
            means=mean(channel)
            mean_values.append(means)
            mmax=max(channel)-min(channel)
            mmax_values.append(mmax)
            var=np.var(channel)
            var_values.append(var)
            app=an.app_entropy(channel, order=2, metric='chebyshev')
            Approximate_E.append(app)

print(len(rms_final))
print(len(perm_entropy_value))
print(len(spectral_entropy_values))
print(len(fd_values))
print(len(detrended_fluctuation_values))
print(len(sample_entropy_values))
print(len(svd_entropy_value))
print(len(Kurtosis_values))
print(len(mean_values))
print(len(mmax_values))
print(len(var_values))
Features_df = {'mean_PSD':mean_psd,'std_psd':std_psd,'Approximate_E':Approximate_E,'perm_entropy_value': perm_entropy_value,'spectral_entropy_values':spectral_entropy_values, 'fd_values':fd_values,'detrended_fluctuation_values':detrended_fluctuation_values,'sample_entropy_values':sample_entropy_values,'svd_entropy_value':svd_entropy_value,'Kurtosis_values':Kurtosis_values,'mean_values':mean_values,'mmax_values':mmax_values,'var_values':var_values,'Labels':Labels}
df = pd.DataFrame(Features_df) 
csv_file = os.path.join(output_folder, 'result_data.csv')
df.to_csv(csv_file, index=False)
Features_df = pd.read_csv('/Users/siddharth/Downloads/EEG Data Plots/result_data.csv')
# Data = Features_df.sample(frac = 1) 
# features = Data[[x for x in Data.columns if x not in ["Label"]]]   # Data for training
# Labels = Data['Labels']                                            # Labels for training
# Labels = Labels.astype('category')
# Corelation_df = features.corr()
# # sns.set(font_scale=0.9)
# # ax = sns.heatmap(Corelation_df, linewidth=0.5,annot=True)
# # plt.show()
# # sns.clustermap(data=Corelation_df, annot=True,linewidth=1,cmap = "Accent",annot_kws={"size": 8},)
# # plt.show()


# splitRatio = 0.3
# train, test = train_test_split(Data ,test_size=splitRatio,
#                                random_state = 123, shuffle = True)

# train_X = train[[x for x in train.columns if x not in ["Labels"]]]   # Data for training
# train_Y = train['Labels']
# test_X = test[[x for x in test.columns if x not in ["Labels"]]]     # Data fo testing
# test_Y = test["Labels"]      
# x_val = train_X[:200]                                                # 50 Sample for Validation
# partial_x_train = train_X[200:]
# partial_x_train = partial_x_train.values
# y_val = train_Y[:200]
# y_val = to_categorical(y_val)
# partial_y_train = train_Y[200:]
# partial_y_train = partial_y_train.values
# partial_y_train = to_categorical(partial_y_train)
# print("Data is prepeared")
# print("Start Building Classifer")












# print("Plots saved in:", output_folder)
# # Create a unique filename for saving the plot
# plot_filename = os.path.splitext(filename)[0] + '_plot.png'
# plot_filepath = os.path.join(output_folder, plot_filename)

# Plot and save the magnitude spectrum
# plt.plot(magnitude_spectrum)
# plt.title(f'Magnitude Spectrum for {filename}')
# plt.savefig(plot_filepath)
# plt.close()  # Close the plot to release resources

# import pandas as pd
# import numpy as np
# from numpy import mean
# from numpy import std
# import matplotlib.pyplot as plt
# import scipy.io
# import scipy.signal
# from scipy.signal import butter,filtfilt,find_peaks,find_peaks, resample
# # from scipy import stats
# from scipy.stats import skew, kurtosis
# from sklearn import preprocessing
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# # from sklearn.metrics import roc_curve,auc, precision_score,recall_score,f1_score
# # from sklearn.feature_selection import SelectKBest, chi2
# # from sklearn.ensemble import ExtraTreesClassifier
# # from keras import layers, models, regularizers
# # import mne
# # from mne import find_events, fit_dipole
# # from autoreject import AutoReject
# import seaborn as sns
# import sys
# import statistics as st
# # from keras.utils import to_categorical
# #import pywt
# import sys
# import antropy as an 
# import h5py
# import time

# import warnings
# warnings.filterwarnings("ignore")


# Channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]   


# T =     150         
# fs =    128         # Sample rate,      Hz
# cutoff = 40          # Desired cutoff frequency of the filter, Hz
# nyq = 0.5 * fs      # Nyquist Frequency
# order = 1           # sin wave can be approx represented as quadratic
# n = int(T * fs)+1   # Total number of samples

# t=np.linspace(0,150,19200)


# Features = ["mean_PSD",         "STD_PSD",
#             "A_mean",           "A_STD",            "A_Var",
#             "A_range",          "A_skew",           "A_kurtosis",
#             "Permutation_E",    "Spectral_E",       "SVD_E",
#             "Approximate_E",    "Sample_E",         "Petrosian_FD",
#             "Katz_FD",          "Higuchi_FD",       
#             "Detrended fluctuation analysis",
#             "Label"] 
#                               # Features' Names

# Features_df = pd.DataFrame(columns = Features)    # Features of Subject 1

# normal_cutoff = [3/nyq ,40/nyq]
# b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)    # Get the filter coefficients
# SS=0
# TIMES = np.zeros([48])

# # Read Data
# start_time = time.time()
# for s in range(1,49):
#     # Data=".\\STEW Dataset\\sub0"+str(s)+"_hi.txt"
#     if s < 10:
#         mat_file = "/Users/siddharth/Downloads/EEG Data Associated with Mental Workload/sub0"+str(s)+"_hi.mat"
#     else:
#         mat_file = "/Users/siddharth/Downloads/EEG Data Associated with Mental Workload/sub"+str(s)+"_hi.mat"

#     with h5py.File(mat_file, 'r') as mat_data:
#         Data = mat_data['EEG'][:]
#     w=0
#     print(Data)
#     for window in range(0,60):
#         EEG_Window = Data[:,w:w+640] # Windowing, Window Len: 5 Sec, Overlap: 2.5 Sec
#         w=w+320
#         Fet = np.zeros([300]) #Temporal Features + Lable array
#         Fet_ch = np.zeros([18])
#         i=0
#         # print(EEG_Window)
#         for i in range(14):
#             channel = EEG_Window[i,:]
#             # Feature Extraction
#     if len(channel) > 0:

#             # PSD Features
#             f, Pxx =scipy.signal.welch(channel,fs) #Extract PSD according to Welch thiorem

#             Fet[i]         = mean(Pxx)                                                      # Mean of PSD
        
#             Fet[i+14]       = std(Pxx)                                                      # Standered Deviation of PSD
        
# #             # Statistics Features
#             Fet[i+28]      = mean(channel)                                                  # Amplitude Mean
        
#             Fet[i+42]      = std(channel)                                                   # Amplitude Standered Deviation
            
#             Fet[i+56]      = np.var(channel)                                                # Amplitude variance
            
#             if len(channel) > 1:
#                 Fet[i+70] = max(channel) - min(channel)  # Amplitude Range
#             else:
#                 Fet[i+70] = 0 
            
#             Fet[i+84]      = skew(channel)                                                  # Amplitude Skew
            
#             Fet[i+98]      = kurtosis(channel)                                              # Amplitude kurtosis
            
# #             # Entropy Features
#             Fet[i+112]      = an.perm_entropy(channel, order=3, normalize=True)                 # Permutation entropy
            
#             Fet[i+126]      = an.spectral_entropy(channel, 100, method='welch', normalize=True) # Spectral entropy
            
#             Fet[i+140]      = an.svd_entropy(channel, order=3, delay=1, normalize=True)         # Singular value decomposition entropy
            
#             Fet[i+154]      = an.app_entropy(channel, order=2, metric='chebyshev')              # Approximate entropy
            
#             Fet[i+168]      = an.sample_entropy(channel, order=2, metric='chebyshev')           # Sample entropy
            
# #             # Fractal dimension Features
#             Fet[i+182]      = an.petrosian_fd(channel)                                          # Petrosian fractal dimension
                
#             Fet[i+196]      = an.katz_fd(channel)                                               # Katz fractal dimension
            
#             if len(channel) > 0:
#                     Fet[i+210] = an.higuchi_fd(channel, kmax=10)  # Higuchi fractal dimension
#             else:
#                     Fet[i+210] = 0  # Set a default value

#                 # Handle Detrended fluctuation analysis to avoid errors
#             if len(channel) > 0:
#                     Fet[i+224] = an.detrended_fluctuation(channel)  # Detrended fluctuation analysis
#             else:
#                     Fet[i+224] = 0  #                            # Detrended fluctuation analysis
            
#     Fet_ch[0]      = mean(Fet[0:14])        # Mean of PSD
#     Fet_ch[1]      = mean(Fet[14:28])       # Standered Deviation of PSD
#     Fet_ch[2]      = mean(Fet[28:42])       # Amplitude Mean
#     Fet_ch[3]      = mean(Fet[42:56])       # Amplitude Standered Deviation
#     Fet_ch[4]      = mean(Fet[56:70])       # Amplitude variance
#     Fet_ch[5]      = mean(Fet[70:84])       # Amplitude Range
#     Fet_ch[6]      = mean(Fet[84:98])       # Amplitude Skew
#     Fet_ch[7]      = mean(Fet[98:112])      # Amplitude kurtosis
#     Fet_ch[8]      = mean(Fet[112:126])     # Permutation entropy
#     Fet_ch[9]      = mean(Fet[126:140])     # Spectral entropy
#     Fet_ch[10]     = mean(Fet[140:154])     # Singular value decomposition entropy
#     Fet_ch[11]     = mean(Fet[154:168])     # Approximate entropy
#     Fet_ch[12]     = mean(Fet[168:182])     # Sample entropy
#     Fet_ch[13]     = mean(Fet[182:196])     # Petrosian fractal dimension
#     Fet_ch[14]     = mean(Fet[196:210])     # Katz fractal dimension
#     Fet_ch[15]     = mean(Fet[210:224])     # Higuchi fractal dimension
#     Fet_ch[16]     = mean(Fet[224:238])     # Detrended fluctuation analysis
#     Fet_ch[17]     = s-1  
#     Features_df.loc[SS]=Fet_ch
#     SS=SS+1

# for i in range(47):
#     TIMES[i] = TIMES[i+1]-TIMES[i]
# print("Mean Feature Extraction Time: ", mean(TIMES[0:47]))
# Features_df.to_csv('extractedFeatures.csv')
# #         Features_df.loc[SS]=Fet_ch
# #         SS=SS+1


# # for i in range(47):
# #     TIMES[i] = TIMES[i+1]-TIMES[i]
# # print("Mean Feature Extraction Time: ", mean(TIMES[0:47]))

# # Features_df.to_csv('extractedFeatures.csv')
# # print("All features were extracted and saved")