import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt

class FreeMusicArchiveMedium(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(annotations_file, delimiter=';')
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    # Load Waveform und Label
    def __getitem__(self, index):
        # Anhand des Dateinamens Track ID, den Pfad herausfinden
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        # Beim Laden konvertieren MP3 → WAV
        signal, sr = torchaudio.load(audio_sample_path, format="mp3")
        # Normalisierungen
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        # Definieren des Wurzelverzeichnisses directory
        directory = self.audio_dir
        # Extrahieren der Track-ID
        track_id = self.annotations.loc[index, 'track_id']
        # Formatieren der Track-ID mit führenden Nullen
        filename = str(track_id).zfill(6)
        # Extrahieren des Segment-Ordners, die ersten drei Zeichen des FIlenames = Name des Unterverzeichnisses
        segment_folder = str(filename)[:3]
        # Pfad zum Audiofile erstellen
        path = os.path.join(directory, segment_folder, filename + '.mp3')
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index]["genre_top"]



if __name__ == "__main__":
    ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
    AUDIO_DIR = 'C:/AI_Datasets/fma_medium/fma_medium'
    # NB_AUDIO_SAMPLES = 1321967
    SAMPLE_RATE = 22500

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
         n_mels=64
    )

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)




    """
    # Plot des ersten Mel-Spektrogramms
    mel_spec, label = fmamed[12545]
    mel_spec = mel_spec.squeeze(0)  # Reduzieren der Kanaldimension
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.log2().detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.xlabel('Zeit in s')
    plt.ylabel(f'Mel filter <= {mel_spectrogram.n_mels}')
    plt.title(f'Mel Spektrogramm für Genre: {label}')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    """

    # Durch das Dataset iterieren und Pfade und Labels ausgeben
    """
    for i in range(len(fmamed)):
        audio_sample_path, label = fmamed._get_audio_sample_path(i), fmamed._get_audio_sample_label(i)
        print("Pfad:", audio_sample_path)
        print("Label:", label)
    """
