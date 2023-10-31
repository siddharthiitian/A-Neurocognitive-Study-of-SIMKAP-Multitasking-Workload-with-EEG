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

lowcut = 12.5  # Lower cutoff frequency in Hz (Beta band)
highcut =  28.5 # Upper cutoff frequency in Hz
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
