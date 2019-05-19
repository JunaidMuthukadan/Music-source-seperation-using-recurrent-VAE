# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:40:13 2019

@author: JuNaiD
"""


import numpy as np
from lxml import etree
import os.path
import librosa

#import Input.Input
from Sample import Sample

def subtract_audio(mix_list, instrument_list):
    '''
    Generates new audio by subtracting the audio signal of an instrument recording from a mixture
    :param mix_list: 
    :param instrument_list: 
    :return: 
    '''

    assert(len(mix_list) == len(instrument_list))
    new_audio_list = list()

    for i in range(0, len(mix_list)):
        new_audio_path = os.path.dirname(mix_list[i]) + os.path.sep + "remainingmix" + os.path.splitext(mix_list[i])[1]
        new_audio_list.append(new_audio_path)

        if os.path.exists(new_audio_path):
            continue
        mix_audio, mix_sr = librosa.load(mix_list[i], mono=False, sr=None)
        inst_audio, inst_sr = librosa.load(instrument_list[i], mono=False, sr=None)
        assert (mix_sr == inst_sr)
        new_audio = mix_audio - inst_audio
        if not (np.min(new_audio) >= -1.0 and np.max(new_audio) <= 1.0):
            print("Warning: Audio for mix " + str(new_audio_path) + " exceeds [-1,1] float range!")

        librosa.output.write_wav(new_audio_path, new_audio, mix_sr) #TODO switch to compressed writing
        print("Wrote accompaniment for song " + mix_list[i])
    return new_audio_list


def create_sample(db_path, instrument_node):
   path = db_path + os.path.sep + instrument_node.xpath("./relativeFilepath")[0].text
   sample_rate = int(instrument_node.xpath("./sampleRate")[0].text)
   channels = int(instrument_node.xpath("./numChannels")[0].text)
   duration = float(instrument_node.xpath("./length")[0].text)
   return Sample(path, sample_rate, channels, duration)

def getDSDFilelist(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    train_vocals, test_vocals, train_bass, test_bass, train_drums, test_drums, train_mixes, test_mixes, train_other, test_other = list(), list(), list(), list(), list(), list(), list(), list(), list(), list()

    for track in tracks:
        # Get mix and vocal instruments
        vocals = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        bass = create_sample(db_path, track.xpath(".//instrument[instrumentName='Bass']")[0])
        drums = create_sample(db_path, track.xpath(".//instrument[instrumentName='Drums']")[0])
        other = create_sample(db_path, track.xpath(".//instrument[instrumentName='Other']")[0])
        #[acc_path] = subtract_audio([mix.path], [vocals.path])
        #acc = Sample(acc_path, vocals.sample_rate, vocals.channels, vocals.duration) # Accompaniment has same signal properties as vocals and mix

        if track.xpath("./databaseSplit")[0].text == "Training":
            train_vocals.append(vocals)
            train_bass.append(bass)
            train_drums.append(drums)
            train_mixes.append(mix)
            train_other.append(other)
        else:
            test_vocals.append(vocals)
            test_bass.append(bass)
            test_drums.append(drums)
            test_mixes.append(mix)
            test_other.append(other)

    dsd_train =np.transpose([train_mixes, train_vocals, train_bass, train_drums, train_other])
    dsd_test  =np.transpose([test_mixes, test_vocals, test_bass, test_drums, test_other])


    return dsd_train,dsd_test



