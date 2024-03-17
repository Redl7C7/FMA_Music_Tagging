import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio


class FreeMusicArchiveMedium(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    # Load Waveform und Label
    def __getitem__(self, index):
        # Anhand der Track ID, den Pfad herausfinden, die Dateien heißen im Vergleich zur CSV noch ".MP3"
        track_id = self.annotations.iloc(index, 'track_id'+'.mp3')
        audio_sample_path = self._get_audio_sample_path(track_id)
        label = self._get_audio_sample_label(track_id)
        # Beim Laden konvertieren MP3 --> WAV
        signal, sr = torchaudio.load(audio_sample_path, format="mp3")
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        # signal = self.transformation(signal)
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

    def _get_audio_sample_path(self, track_id):
        # Es gibt über 150 Unterordner, die nach track_id gescannt werden müssen
        # Wir definieren das Wurzelverzeichnis directory
        directory = self.audio_dir
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename == track_id:
                    path = os.path.join(dirpath, filename)
                return path

    def _get_audio_sample_label(self, index):
        # If-Abfrage, ob es genau 1 Hauptgenre gibt
        track_genre_top = self.annotations.iloc[index, 5]
        if isinstance(track_genre_top, str):
            self.annotations.loc[index, 5] = torch.tensor(int(track_genre_top))
        else:
            label = None
        return self.annotations.loc[index, 5]


if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:/AI_Datasets/fma_medium/genres.csv"
    AUDIO_DIR = "C:/AI_Datasets/fma_medium/fma_medium"
    NB_AUDIO_SAMPLES = 1321967
    SAMPLE_RATE = 44100

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE, AUDIO_DIR, SAMPLE_RATE)
    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = fmamed[0]