import librosa
import numpy as np
from config import ModelConfig
import soundfile as sf

# Batch considered
def get_random_wav(filenames, sec, sr=ModelConfig.SR):
    mixes =[]
    vocals =[]
    bass =[]
    drums =[]
    other =[]
    for file in filenames:
        duration_ = int(file[0].duration)
        start = np.random.choice(range(np.maximum(0, duration_ - sec)), 1)[0]
        mixes_ = librosa.load(file[0].path, sr = sr,offset = start, duration =sec, mono = True)[0]
        vocals_ = librosa.load(file[1].path, sr = sr,offset = start, duration =sec, mono = True)[0]
        bass_= librosa.load(file[2].path, sr = sr,offset = start, duration = sec, mono = True)[0]
        drums_ = librosa.load(file[3].path, sr = sr,offset = start, duration =sec, mono = True)[0]
        other_ = librosa.load(file[4].path, sr = sr,offset = start, duration =sec, mono = True)[0]
        mixes.append(mixes_)
        vocals.append(vocals_)
        bass.append(bass_)
        drums.append(drums_)
        other.append(other_)
    return mixes, drums

# Batch considered
def to_spectrogram(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav)))

# Batch considered
def to_wav(mag, phase, len_hop=ModelConfig.L_HOP):
    stft_matrix = get_stft_matrix(mag, phase)
    return np.array(list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_matrix)))

# Batch considered
def to_wav_from_spec(stft_maxrix, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda s: librosa.istft(s, hop_length=len_hop), stft_maxrix)))

# Batch considered
def to_wav_mag_only(mag, init_phase, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP, num_iters=50):
    #return np.array(list(map(lambda m_p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p)[0], list(zip(mag, init_phase))[1])))
    return np.array(list(map(lambda m: lambda p: griffin_lim(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p), list(zip(mag, init_phase))[1])))

# Batch considered
def get_magnitude(stft_matrixes):
    return np.abs(stft_matrixes)

# Batch considered
def get_phase(stft_maxtrixes):
    return np.angle(stft_maxtrixes)

# Batch considered
def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)

# Batch considered
def soft_time_freq_mask(target_src, remaining_src):
    mask = np.abs(target_src) / (np.abs(target_src) + np.abs(remaining_src) + np.finfo(float).eps)
    return mask

# Batch considered
def hard_time_freq_mask(target_src, remaining_src):
    mask = np.where(target_src > remaining_src, 1., 0.)
    return mask

def write_wav(data, path, sr=ModelConfig.SR, format='wav', subtype='PCM_16'):
    sf.write('{}.wav'.format(path), data, sr, format=format, subtype=subtype)

def griffin_lim(mag, len_frame, len_hop, num_iters, phase_angle=None, length=None):
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = get_stft_matrix(mag, phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=len_frame, hop_length=len_hop, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=len_frame, win_length=len_frame, hop_length=len_hop)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = get_stft_matrix(mag, phase_angle)
    return wav

    # shape = (batch_size, n_freq, n_frames) => (batch_size, n_frames, n_freq)
def spec_to_batch(src):
    num_wavs, freq, n_frames = src.shape

        # Padding
    pad_len = 0
    if n_frames % ModelConfig.SEQ_LEN > 0:
        pad_len = (ModelConfig.SEQ_LEN - (n_frames % ModelConfig.SEQ_LEN))
    pad_width = ((0, 0), (0, 0), (0, pad_len))
    padded_src = np.pad(src, pad_width=pad_width, mode='constant', constant_values=0)

    assert(padded_src.shape[-1] % ModelConfig.SEQ_LEN == 0)

    batch = np.reshape(padded_src.transpose(0, 2, 1), (-1, ModelConfig.SEQ_LEN, freq))
    return batch, padded_src