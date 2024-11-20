import os
import pickle as pkl
from os import listdir
from os.path import isfile, join

import librosa
import opensmile
import pandas as pd
from opensmile import Smile
from sklearn.neural_network import MLPClassifier


def loadVoice(voiceFile):
    audio = librosa.load(voiceFile, sr=16000, res_type="kaiser_best")[0]
    return audio


def trimVoice(voiceFile):
    voice_t, index = librosa.effects.trim(voiceFile)
    return voice_t


def getFeatures(audio, fileName):
    smile = Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    features = smile.process_signal(audio, 16000)
    features.index = [fileName]
    return features


if __name__ == '__main__':
    dir = "src\\resource\\voices"
    labelMap = {1: "Angry", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Neutral", 6: "Surprise"}
    genderMap = {1: "Male", 2: "Female"}
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    audios = []
    for file in onlyfiles:
        audio = loadVoice(os.path.join(dir, file))
        audio_t = trimVoice(audio)
        audios.append(audio_t)
    audios_features = []
    for i in range(len(audios)):
        features = getFeatures(audios[i], onlyfiles[i])
        audios_features.append(features)
    audios_features_dataframe = pd.concat(audios_features)
    clf = ""
    with open(f'src/resource/label_clf.pkl', "rb") as f:
        clf: MLPClassifier = pkl.load(f)
        f.close()
    pred_label = clf.predict(audios_features_dataframe)
    prob_label = clf.predict_proba(audios_features_dataframe)

    with open(f'src/resource/gender_clf.pkl', "rb") as f:
        clf: MLPClassifier = pkl.load(f)
        f.close()
    pred_gender = clf.predict(audios_features_dataframe)
    prob_gender = clf.predict_proba(audios_features_dataframe)
    for i in range(len(audios_features_dataframe)):
        print(audios_features_dataframe.index[i], "::: prediction: ", labelMap[pred_label[i]], "::: probablity => ", "Angry:", prob_label[i][0],
              " Fear:", prob_label[i][1], " Happy:", prob_label[i][2], " Sad:", prob_label[i][3], " Neutral:", prob_label[i][4], " Surprise:",
              prob_label[i][5])

        print(audios_features_dataframe.index[i], "::: prediction: ", genderMap[pred_gender[i]], "::: probablity => ", " Male:", prob_gender[i][0],
              " Female:", prob_gender[i][1])
