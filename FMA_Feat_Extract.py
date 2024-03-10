import os

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torchaudio.utils import download_asset

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd
import matplotlib.patches


#Seeds sind Randomizer für numerische Werte, wird benötigt um Daten durchzumischen oder initiale Gewqichtungen zu vergeben
torch.random.manual_seed(0)

#Audiosignale laden
AUDIO_DIR = os.environ.get('E:/Neuer Ordner/fma_full/fma_full')

#Funktion: Waveform zur Visualisierung des Signals
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


#Multiskalenanalyse des Audiosignals/ Musikstücks
#Multiskalenanalyse ist ein Spektogramm mit verschiedener Intervallgröße oder auch N für die Fast Fourier Transformation (FFT)
#Es wird daher drei Auflösungen für diese Arbeit geben N=512, N=1024 und N=2048

#Step 1 - Spektogramm generisch definieren
#Funktion: generisches Spektogramm
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


