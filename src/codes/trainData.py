import os
import pickle
import random
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
    dir1 = "src\\resource\\male\\male"
    dir2 = "src\\resource\\female\\female"
    labelMap = {1: "Angry", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Neutral", 6: "Surprise"}
    labelReverseMap = {"A": 1, "F": 2, "H": 3, "S": 4, "N": 5, "W": 6}
    genderMap = {1: "Male", 2: "Female"}
    genderReverseMap = {"M": 1, "F": 2}
    onlyfiles1 = [os.path.join(dir1, f) for f in listdir(dir1) if isfile(join(dir1, f))]
    onlyfiles2 = [os.path.join(dir2, f) for f in listdir(dir2) if isfile(join(dir2, f))]
    onlyfiles = []
    onlyfiles.extend(onlyfiles1)
    onlyfiles.extend(onlyfiles2)
    random.shuffle(onlyfiles)
    # onlyfiles = onlyfiles[:100]
    audios = []
    for file in onlyfiles:
        audio = loadVoice(file)
        audio_t = trimVoice(audio)
        audios.append(audio_t)
    audios_features = []
    for i in range(len(audios)):
        features = getFeatures(audios[i], onlyfiles[i])
        audios_features.append(features)
    audios_features_dataframe = pd.concat(audios_features)
    labels_features = []
    gender_features = []
    for i in range(len(audios_features_dataframe)):
        labels_features.append(labelReverseMap[audios_features_dataframe.index[i].split("\\")[-1][3]])
        gender_features.append(genderReverseMap[audios_features_dataframe.index[i].split("\\")[-1][0]])

    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 70, 30, 15,), learning_rate='adaptive', max_iter=100000000,
                        learning_rate_init=0.0001)
    clf.fit(audios_features_dataframe, labels_features)
    pickle.dump(clf, open("src/resource/label_clf.pkl", "wb"))

    clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(100, 70, 30, 15,), learning_rate='adaptive', max_iter=100000000,
                        learning_rate_init=0.0001)
    clf.fit(audios_features_dataframe, gender_features)
    pickle.dump(clf, open("src/resource/gender_clf.pkl", "wb"))
