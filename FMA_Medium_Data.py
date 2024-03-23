import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class FreeMusicArchiveMedium(Dataset):
    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file, delimiter=';')
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    # Load Waveform und Label
    def __getitem__(self, index):
        # Anhand des Dateinamens Track ID, den Pfad herausfinden
        track_id = self.annotations.iloc[index, 1]
        audio_sample_path = self._get_audio_sample_path(track_id)
        label = self._get_audio_sample_label(track_id)
        # Beim Laden konvertieren MP3 → WAV
        signal, sr = torchaudio.load(audio_sample_path, format="mp3")
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

    def _get_audio_sample_path(self, track_id):
        # Es gibt über 150 Unterordner, die nach track_id gescannt werden müssen
        # definieren des Wurzelverzeichnisses directory
        directory = self.audio_dir
        for dirpath, dirname, filenames in os.walk(directory):
            for filename in filenames:
                if filename == track_id:
                    path = os.path.join(dirpath, filename)
                return path

    def _get_audio_sample_label(self, index):
        # If-Abfrage, ob es genau 1 Hauptgenre gibt
        track_genre_top = self.annotations.iloc[index, 4]
        if isinstance(track_genre_top, str):
            self.annotations.iloc[index, 4] = torch.tensor(int(track_genre_top))
        return self.annotations.iloc[index, 4]


if __name__ == "__main__":
    ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
    AUDIO_DIR = 'C:/AI_Datasets/fma_medium/fma_medium'
    NB_AUDIO_SAMPLES = 1321967
    SAMPLE_RATE = 44100

fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE, AUDIO_DIR)
fmamed.__init__(ANNOTATIONS_FILE, AUDIO_DIR)
print(f"Datensatz hat {len(fmamed)} Datensätze.")
signal, label = fmamed[0]
