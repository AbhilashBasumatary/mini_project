##packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
import seaborn as sn
import numpy as np
import pandas as pd
print( ' Running')
def make_predictions(pa):

    df = pd.read_csv('Data/features_30_sec.csv')

    #drop unnecessary columns

    x = df.drop(['filename','length','harmony_mean','harmony_var','label'],axis = 'columns')
    y = df.label


    #split train_test_split
    x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



    #Decision Tree model training
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(x_train,y_train)

    model.predict(x_test)

    ## feature extraction funciton



    def zero_cross(x):
        zero_crossings = librosa.feature.zero_crossing_rate(x)
        return sum(zero_crossings)

    def spec_center(x, sr):
        spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
        return spectral_centroids

    def spec_rollof(x,sr):
        spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
        return spectral_rolloff

    def chroma_feature(x,sr):
        chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=512)
        return chromagram
    def zero_crossing_rate(x,sr):
        zero = librosa.feature.zero_crossing_rate(x)
        return zero
    def spectral_bandwith(x,sr):
        spec = librosa.feature.spectral_bandwidth(x,sr)
        return spec;
    def tempo(x,sr):
        onset_env = librosa.onset.onset_strength(y=x, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return tempo[0];
    def perceptor(y,sr):
        C = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1')))
        freqs = librosa.cqt_frequencies(C.shape[0],
        fmin=librosa.note_to_hz('A1'))
        perceptual_CQT = librosa.perceptual_weighting(C**2,freqs,ref=np.max)
        return perceptual_CQT
    def mfcc(y,sr):
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        return mfccs;
    def rms(y):
        rms_val=librosa.feature.rms(y=y)
        return rms_val;


    def extract_features(x,sr):
        df2 = pd.DataFrame([[chroma_feature(x,sr).mean(),chroma_feature(x,sr).var(),rms(x).mean(),
                        rms(x).var(),spec_center(x,sr).mean(),spec_center(x,sr).var(),spectral_bandwith(x,sr).mean(),
                         spectral_bandwith(x,sr).var(),spec_rollof(x,sr).mean(),spec_rollof(x,sr).var(),zero_crossing_rate(x,sr).mean(),
                         zero_crossing_rate(x,sr).var(),perceptor(x,sr).mean(),perceptor(x,sr).var(),tempo(x,sr)
                        ]], columns=['chroma_stft_mean','chroma_stft_var','rms_mean','rms_var',
                                          'spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean',
                                          'spectral_bandwidth_var','rolloff_mean','rolloff_var','zero_crossing_rate_mean',
                                          'zero_crossing_rate_var','perceptr_mean','perceptr_var','tempo'])
        mfccs = mfcc(x,sr);
        n=0;
        for i in mfccs:
            df2['mfcc'+str(n+1)+'_mean']=mfccs[n].mean();
            df2['mfcc'+str(n+1)+'_var']=mfccs[n].var();
            n = n+1;
        return df2;

    x,sr =librosa.load(pa)
    new_df = extract_features(x,sr);
    new_df

    return model.predict(new_df);