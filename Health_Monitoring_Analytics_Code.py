import numpy as np 

import librosa 

import librosa.display 

import scipy.signal as signal 

import matplotlib.pyplot as plt 

import pandas as pd 

file_path = "C:/Users/User/Documents/Predictive Modeling and Decision Making" 

files = librosa.util.find_files(file_path, ext=['mp3','m4a','wav'])  

files = np.asarray(files) 

# Short-Time Fourier Transform (STFT) for Time-Frequency Analysis 

def compute_stft(audio_data, sampling_rate, n_fft=2048, hop_length=512): 

    stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length) 

    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max) 

    return stft_db 

  

# Average Airflow Volume (based on amplitude) 

def calculate_average_airflow(audio_data, sampling_rate): 

    amplitude_envelope = np.abs(audio_data) 

    avg_airflow_volume = np.mean(amplitude_envelope) 

    return avg_airflow_volume 

  

#  Breathing Frequency (Breaths per minute) 

def calculate_breathing_frequency(audio_data, sampling_rate, window_size=3): 

    # Low-pass filter to isolate breathing sounds (0.2-1.5 Hz typical range) 

    b, a = signal.butter(2, [0.2, 1.5], btype='bandpass', fs=sampling_rate)  

    filtered_audio = signal.filtfilt(b, a, audio_data) 

     

    # Detect peaks in the filtered signal (representing breathing cycles) 

    peaks, _ = signal.find_peaks(filtered_audio, height=np.mean(filtered_audio), distance=sampling_rate*2) 

    breath_count = len(peaks) 

     

    # Convert breath count to breaths per minute 

    audio_duration = len(audio_data) / sampling_rate 

    breathing_frequency = (breath_count / audio_duration) * 60 

    return breathing_frequency 

  

# Breath Strength (Peak Intensity) 

def calculate_breath_strength(audio_data, sampling_rate): 

    amplitude_envelope = np.abs(audio_data) 

    breath_strength = np.max(amplitude_envelope) 

    return breath_strength 

  

#  Cough Detection (Frequency and Intensity) 

def detect_cough_events(audio_data, sampling_rate, threshold=0.6): 

    # Band-pass filter for cough sound detection (frequencies 300-6000 Hz) 

    b, a = signal.butter(2, [300, 6000], btype='bandpass', fs=sampling_rate)#300, 6k is the range of coughing sound frequency, need to check the range. 

    filtered_audio = signal.filtfilt(b, a, audio_data) 

     

    # Detect cough peaks (above a certain amplitude threshold) 

    peaks, properties = signal.find_peaks(filtered_audio, height=threshold) 

    cough_frequency = len(peaks) 

    cough_intensity = np.mean(properties['peak_heights']) if cough_frequency > 0 else 0 

    return cough_frequency, cough_intensity 

  

# 7. Background Noise Level (Signal-to-Noise Ratio) 

def calculate_snr(audio_data, sampling_rate, noise_threshold=0.1): 

    # Separate background noise from signal by identifying silent regions 

    rms_energy = librosa.feature.rms(y = audio_data) 

    noise_level = np.mean(rms_energy[rms_energy < noise_threshold]) 

    signal_level = np.mean(rms_energy[rms_energy > noise_threshold]) 

     

    snr = 10 * np.log10(signal_level / noise_level) if noise_level > 0 else np.inf 

    return snr 

  

# Plotting the waveform and STFT 

def plot_waveform_and_spectrogram(audio_data, sampling_rate, stft_db): 

    # Plot the waveform 

    plt.figure(figsize=(12, 6)) 

    plt.subplot(2, 1, 1) 

    librosa.display.waveshow(audio_data, sr=sampling_rate) 

    plt.title("Waveform") 

     

    # Plot the spectrogram 

    plt.subplot(2, 1, 2) 

    librosa.display.specshow(stft_db, sr=sampling_rate, hop_length=512, x_axis='time', y_axis='log') 

    plt.colorbar(format="%+2.0f dB") 

    plt.title("Spectrogram (STFT)") 

    plt.tight_layout() 

    plt.show() 

  

# Function to normalize variables based on healthy, mild, moderate, severe ranges 

def normalize_variable(value, healthy_range, mild_range, moderate_range, severe_range): 

    if healthy_range[0] <= value <= healthy_range[1]: 

        return 1.0 

    elif mild_range[0] <= value <= mild_range[1]: 

        return 0.75 + 0.25 * (value - mild_range[0]) / (mild_range[1] - mild_range[0]) 

    elif moderate_range[0] <= value <= moderate_range[1]: 

        return 0.5 + 0.25 * (value - moderate_range[0]) / (moderate_range[1] - moderate_range[0]) 

    elif severe_range[0] <= value <= severe_range[1]: 

        return 0.25 * (value - severe_range[0]) / (severe_range[1] - severe_range[0]) 

    else: 

        return 0  # If value is out of range, treat it as severe 

  

# Define the ranges for normalization (based on earlier categorization) 

def get_normalized_scores(avg_airflow_volume, breathing_frequency, breath_strength, cough_frequency, cough_intensity, snr, 

                          inhalation_exhalation_ratio, spectral_centroid, spectral_bandwidth): 

    normalized_airflow = normalize_variable(avg_airflow_volume, [0.5, 1.0], [0.4, 0.5], [0.3, 0.4], [0, 0.3]) 

    normalized_breath_freq = normalize_variable(breathing_frequency, [12, 20], [20, 25], [25, 30], [30, 60]) 

    normalized_breath_strength = normalize_variable(breath_strength, [0.7, 1.0], [0.6, 0.7], [0.4, 0.6], [0, 0.4]) 

    normalized_cough_freq = normalize_variable(cough_frequency, [0, 3], [4, 6], [6, 10], [10, 60]) 

    normalized_cough_intensity = normalize_variable(cough_intensity, [0.5, 1.0], [0.4, 0.5], [0.2, 0.4], [0, 0.2]) 

    #normalized_cough_intensity = normalize_variable(cough_intensity, [0.0, 0.5], [0.5, 0.7], [0.7, 0.8], [0.8, 1.0]) 

    normalized_snr = normalize_variable(snr, [15, np.inf], [12, 15], [8, 12], [0, 8]) 

    normalized_inhal_exhal_ratio = normalize_variable(inhalation_exhalation_ratio, [0.8, 1.2], [0.6, 0.8], [0.4, 0.6], [0, 0.4]) 

    normalized_spectral_centroid = normalize_variable(spectral_centroid, [1000, 1500], [700, 1000], [500, 700], [0, 500]) 

    normalized_spectral_bandwidth = normalize_variable(spectral_bandwidth, [1500, 3000], [1000, 1500], [500, 1000], [0, 500]) 

  

    return (normalized_airflow, normalized_breath_freq, normalized_breath_strength, normalized_cough_freq, 

            normalized_cough_intensity, normalized_snr, normalized_inhal_exhal_ratio, 

            normalized_spectral_centroid, normalized_spectral_bandwidth) 

  

# Health metric function based on the weighted sum of normalized variables 

def health_metric(avg_airflow_volume, breathing_frequency, breath_strength, cough_frequency, cough_intensity, snr, 

                  inhalation_exhalation_ratio, spectral_centroid, spectral_bandwidth): 

    # Define weights 

     weights = { 

        'breathing_frequency': 0.20, 

        'avg_airflow_volume': 0.15, 

        'breath_strength': 0.15, 

        'cough_frequency': 0.10, 

        'cough_intensity': 0.10, 

        'snr': 0.05, 

        'inhal_exhal_ratio': 0.10, 

        'spectral_centroid': 0.075, 

        'spectral_bandwidth': 0.075 

    } 

  

    # Get normalized values 

    (norm_airflow, norm_breath_freq, norm_breath_strength, norm_cough_freq, norm_cough_intensity, norm_snr, 

     norm_inhal_exhal_ratio, norm_spectral_centroid, norm_spectral_bandwidth) = get_normalized_scores( 

        avg_airflow_volume, breathing_frequency, breath_strength, cough_frequency, cough_intensity, snr, 

        inhalation_exhalation_ratio, spectral_centroid, spectral_bandwidth 

    ) 

  

    # Calculate health score 

    score = (weights['breathing_frequency'] * norm_breath_freq + 

             weights['avg_airflow_volume'] * norm_airflow + 

             weights['breath_strength'] * norm_breath_strength + 

             weights['cough_frequency'] * norm_cough_freq + 

             weights['cough_intensity'] * norm_cough_intensity + 

             weights['snr'] * norm_snr + 

             weights['inhal_exhal_ratio'] * norm_inhal_exhal_ratio + 

             weights['spectral_centroid'] * norm_spectral_centroid + 

             weights['spectral_bandwidth'] * norm_spectral_bandwidth) 

  

    # Categorize the health based on the score 

    if score > 0.75: 

        category = "Healthy" 

    elif 0.5 <= score <= 0.75: 

        category = "Mild Risk" 

    elif 0.25 <= score < 0.5: 

        category = "Moderate Risk" 

    else: 

        category = "Severe Risk" 

     

    return score, category 

  

  

# Inhalation-to-Exhalation Ratio 

def calculate_inhalation_exhalation_ratio(audio_data, sampling_rate, threshold=0.02): 

    audio_data_abs = np.abs(audio_data) 

    inhale_exhale_periods = audio_data_abs > threshold  # Identify regions with sound 

    changes = np.diff(inhale_exhale_periods.astype(int))  # Detect changes (start and end) 

    inhale_times = np.where(changes == 1)[0] / sampling_rate 

    exhale_times = np.where(changes == -1)[0] / sampling_rate 

    inhale_duration = np.sum(np.diff(inhale_times)) 

    exhale_duration = np.sum(np.diff(exhale_times)) 

    ratio = inhale_duration / (exhale_duration if exhale_duration != 0 else 1) 

    return ratio 

  

# Formant Frequencies (simplified estimation using FFT) 

def calculate_formant_frequencies(audio_data, sampling_rate, formant_count=2): 

    fft_spectrum = np.fft.fft(audio_data) 

    freqs = np.fft.fftfreq(len(fft_spectrum), 1 / sampling_rate) 

    peaks = np.argsort(np.abs(fft_spectrum))[-formant_count:]  # Top peaks for formants 

    formants = np.abs(freqs[peaks]) 

    return sorted(formants) 

  

# Spectral Centroid 

def calculate_spectral_centroid(audio_data, sampling_rate): 

    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate) 

    return np.mean(spectral_centroid) 

  

# Zero-Crossing Rate 

def calculate_zero_crossing_rate(audio_data): 

    zcr = librosa.feature.zero_crossing_rate(audio_data) 

    return np.mean(zcr) 

  

# Spectral Bandwidth 

def calculate_spectral_bandwidth(audio_data, sampling_rate): 

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sampling_rate) 

    return np.mean(spectral_bandwidth) 

AverageAirflowVolume = [] 

BreathingFrequency = [] 

BreathStrength = [] 

CoughFrequency = [] 

CoughIntensity = [] 

SNR = [] 

Score = [] 

Category = [] 

Inhalation_exhalation_ratio = [] 

Spectral_centroid = [] 

Spectral_bandwidth = [] 

def calculate_terms(audio_data, sampling_rate): 

    # Compute STFT for time-frequency analysis 

    stft_db = compute_stft(audio_data, sampling_rate) 

    score = 0 

     

    # Calculate variables 

    avg_airflow_volume = calculate_average_airflow(audio_data, sampling_rate) 

    breathing_frequency = calculate_breathing_frequency(audio_data, sampling_rate) 

    breath_strength = calculate_breath_strength(audio_data, sampling_rate) 

    cough_frequency, cough_intensity = detect_cough_events(audio_data, sampling_rate) 

    snr = calculate_snr(audio_data, sampling_rate) 

    inhalation_exhalation_ratio = calculate_inhalation_exhalation_ratio(audio_data, sampling_rate) 

    formant_frequencies = calculate_formant_frequencies(audio_data, sampling_rate) 

    spectral_centroid = calculate_spectral_centroid(audio_data, sampling_rate) 

    zcr = calculate_zero_crossing_rate(audio_data) 

    spectral_bandwidth = calculate_spectral_bandwidth(audio_data, sampling_rate) 

    score, category = health_metric(avg_airflow_volume, breathing_frequency, breath_strength, cough_frequency, cough_intensity, snr, 

                                    inhalation_exhalation_ratio, spectral_centroid, spectral_bandwidth) 

  

     

    # Plot waveform and spectrogram 

    plot_waveform_and_spectrogram(audio_data, sampling_rate, stft_db) 

      

    AverageAirflowVolume.append(avg_airflow_volume) 

    BreathingFrequency.append(breathing_frequency) 

    BreathStrength.append(breath_strength) 

    CoughFrequency.append(cough_frequency) 

    CoughIntensity.append(cough_intensity) 

    SNR.append(snr) 

    Score.append(score) 

    Category.append(category) 

    Inhalation_exhalation_ratio.append(inhalation_exhalation_ratio) 

    Spectral_centroid.append(spectral_centroid) 

    Spectral_bandwidth.append(spectral_bandwidth) 

pulmonary_data = pd.DataFrame({'Average Airflow Volume':AverageAirflowVolume,  

                       'Breathing Frequency':BreathingFrequency,  

                       'Breath Strength': BreathStrength,  

                       'Cough Frequency':CoughFrequency, 

                       'Cough Intensity':CoughIntensity, 

                       'SNR' : SNR, 

                       'Inhalation Exhalation Ratio': Inhalation_exhalation_ratio, 

                       'Spectral_centroid':Spectral_centroid, 

                       'Spectral_bandwidth':Spectral_bandwidth, 

                       'Score' : Score, 

                       'Category' : Category, 

                      }) 

  

# writing to Excel 

datatoexcel = pd.ExcelWriter('variables12112024.xlsx') 

  

# write DataFrame to excel 

pulmonary_data.to_excel(datatoexcel) 

# save the Excel 

data to excel.close() 

 

Data Analysis Code: 

import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.preprocessing import StandardScaler 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

from sklearn.svm import SVC 

from sklearn.ensemble import GradientBoostingClassifier 

 

data = pd.read_excel("C:\\Users\\User\\Documents\\Predictive Modeling and Decision Making\\Final\\Final.xlsx")  

data = data.fillna(0) 

data["Gender"].replace(['Male', 'Female'], [0, 1], inplace=True) 

data["Smoking_History"].replace(['Yes', 'No'], [1, 0], inplace=True) 

data["Covid_Symptoms"].replace(['Positive', 'Negative'], [1, 0], inplace=True)  

y = data["Category"]   

x = data.drop("Category", axis=1)  # Features (exclude the target column 'label')  

x = x.fillna(0) 

# Dividing the data into train and test 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

# Feature Scaling (Standardize the features for better performance) 

scaler = StandardScaler() 

x_train_scaled = scaler.fit_transform(x_train) 

x_test_scaled = scaler.transform(x_test) 

 

# Random Forest Classifier 

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) 

rf_clf.fit(x_train_scaled, y_train) 

 

# Support Vector Machine (SVM) 

svm_clf = SVC(kernel='linear', random_state=42) 

svm_clf.fit(x_train_scaled, y_train) 

 

# Gradient Boosting Classifier 

gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42) 

gb_clf.fit(x_train_scaled, y_train) 

 

# Evaluate Models on Test Data 

# Random Forest 

y_pred_rf = rf_clf.predict(x_test_scaled) 

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf)) 

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf)) 

print("Classification Report:\n", classification_report(y_test, y_pred_rf)) 

  

# Support Vector Machine 

y_pred_svm = svm_clf.predict(x_test_scaled) 

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm)) 

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm)) 

print("Classification Report:\n", classification_report(y_test, y_pred_svm)) 

  

# Gradient Boosting 

y_pred_gb = gb_clf.predict(x_test_scaled) 

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb)) 

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gb)) 

print("Classification Report:\n", classification_report(y_test, y_pred_gb)) 

 

# Cross-Validation to check model robustness 

print("Cross-Validation Random Forest:", cross_val_score(rf_clf, x_train_scaled, y_train, cv=5).mean()) 

print("Cross-Validation SVM:", cross_val_score(svm_clf, x_train_scaled, y_train, cv=5).mean()) 

print("Cross-Validation Gradient Boosting:", cross_val_score(gb_clf, x_train_scaled, y_train, cv=5).mean()) 