import os
import torch
import torchaudio
import numpy
import torchaudio.functional as F
import torchaudio.transforms as T
from torchaudio.utils import download_asset

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import matplotlib.patches

# Seeds sind Randomizer für numerische Werte, wird benötigt um Daten durchzumischen oder initiale Gewichtungen zu vergeben
torch.random.manual_seed(0)

# Audiosignale Pfadangabe
AUDIO_DIR: str = r"E:/Neuer Ordner/fma_full/fma_full"
# content_dir: List[str] = os.listdir(AUDIO_DIR)


# Funktion: Waveform zur Visualisierung des Signals
def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


# Multi Skalen Analyse des Audiosignals/ Musikstücks
# Multi Skalen Analyse ist ein Spektogramm mit verschiedener Intervallgröße oder auch N oder n_fft für die Fast Fourier Transformation (FFT)
# Es wird daher drei Auflösungen für diese Arbeit geben N=512, N=1024 und N=2048

# Step 1 - Spektogramm generisch definieren
# Funktion: generisches Spektogramm
def plot_spectrogram(specgram, title=None, ylabel="Frequenzbereich", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

# Mel-Filter Bank
def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")

print("Ab hier Plotten")
# Test der Plots bis hier
MUSIC_WAVEFORM, SAMPLE_RATE = torchaudio.load(os.sep.join([AUDIO_DIR, '/063/063012.mp3']))
# Transformation bestimmen --> Spektogramm mit N=512 Samples
spectrogram = T.Spectrogram(n_fft=512)

# Transformation durchführen
spec = spectrogram(MUSIC_WAVEFORM)

# Plotten
fig, axs = plt.subplots(2, 1)
plt.show(plot_waveform(MUSIC_WAVEFORM, SAMPLE_RATE, title="Original Waveformat", ax=axs[0]))
plt.show(plot_spectrogram(spec[0], title="Spektogramm", ax=axs[1]))
fig.tight_layout()
print("Eigentlich fertig...hier...")


