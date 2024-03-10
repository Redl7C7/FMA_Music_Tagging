import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd

# Seeds sind Randomizer für numerische Werte, wird benötigt um Daten zu mischen oder initiale Gewichtungen zu vergeben
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
MUSIC_MP3 = r"E:/Neuer Ordner/fma_full/fma_full/063/063012.mp3"
MUSIC_WAVEFORM, SAMPLE_RATE = torchaudio.load(MUSIC_MP3, format="mp3")
# Transformation bestimmen --> Spektogramm mit N=512 Samples
spectrogram512 = T.Spectrogram(n_fft=512)
spectrogram1024 = T.Spectrogram(n_fft=1024)
spectrogram2048 = T.Spectrogram(n_fft=2048)

# Transformation durchführen
spec512 = spectrogram512(MUSIC_WAVEFORM)
# spec1024 = spectrogram1024(MUSIC_WAVEFORM)
# spec2048 = spectrogram2048(MUSIC_WAVEFORM)

# Konvertiere Sample-Indizes in Zeit
time_signal = librosa.samples_to_time(np.arange(len(MUSIC_WAVEFORM)), sr=SAMPLE_RATE)

# Plotten
fig, axs = plt.subplots(2, 1)
print("Waveform:")
plot_waveform(MUSIC_WAVEFORM, SAMPLE_RATE, title="Original Waveformat", ax=axs[0])
plot_spectrogram(spec512[0], title="Spektogramm Auflösung N=512", ax=axs[1])
fig.tight_layout()

# Multiskalenauflösung
n_ffts = [512, 1024, 2048]
hop_length = 128

specs = []
for n_fft in n_ffts:
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    spec = spectrogram(MUSIC_WAVEFORM)
    specs.append(spec)

# MSA Plotten
fig, axs = plt.subplots(len(specs), 1, sharex=True)
for i, (spec, n_fft) in enumerate(zip(specs, n_ffts)):
    plot_spectrogram(spec[0], ylabel=f"n_fft={n_fft}", ax=axs[i])
    axs[i].set_xlabel(None)
fig.tight_layout()


