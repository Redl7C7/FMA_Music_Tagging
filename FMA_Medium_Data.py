import os
import torch
from torch.utils.data import Dataset
import torchaudio.utils.ffmpeg_utils
import pandas as pd
import torchaudio


# import Genre_Classifier
# import matplotlib.pyplot as plt


class FreeMusicArchiveMedium(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file, delimiter=';')  # Semikolon als Trennzeichen
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    # Load Waveform und Label
    def __getitem__(self, index):
        # Anhand des Dateinamens Track ID, den Pfad herausfinden
        audio_sample_path = self._get_audio_sample_path(index)
        print(f"Folgender Song: {audio_sample_path}")
        label = self._get_audio_sample_label(index)
        # Beim Laden konvertieren MP3 → WAV
        signal, sr = torchaudio.load(audio_sample_path)
        # signal, sr = torchaudio.load()
        """
        if signal is None:
            print("Fehler beim Laden der Audiodatei.")
            # Füge hier weitere Fehlerbehandlung hinzu, falls erforderlich
        else:
            print("Audiodaten erfolgreich geladen.")
        """
        # Überprüfe die Abtastrate
        # print("Abtastrate (sr):", sr)

        # Überprüfe die Form der Audiodaten
        # print("Form der Audiodaten (Signal):", signal.shape)
        # Normalisierungen
        # gleiche Sample-RATE
        signal = self._resample_if_necessary(signal, sr)
        # print("resample: Form der Audiodaten (Signal):", signal.shape)
        # eindimensionale Eingabe (1 Kanal)
        signal = self._mix_down_if_necessary(signal)
        # print("mix down: Form der Audiodaten (Signal):", signal.shape)
        # Cut, wenn Song zu lang
        signal = self._cut_if_necessary(signal)
        # print(f"cut: Form der Audiodaten (Signal):{signal.shape}")
        # Zero Right Padding für kürzere Songs
        signal = self._right_pad_if_necessary(signal)
        # print("pad: Form der Audiodaten (Signal):", signal.shape)
        # print("Form der Audiodaten (Signal):", signal.shape)
        # Normalisierung auf den Bereich [-1, 1]
        signal = self.transformation(signal)
        return signal, label

    # Die Labels werden aktuell als String gespeichert
    # Daher muss ein INT-Wert je Genre für die Klassifikation zugeordnet werden:

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)  # (left Padding Anz, right Padding Anz)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            print(f"Samplerate original: {sr}")
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] == 2:
            signal = torch.mean(signal, dim=0, keepdim=True)
        # print("Form des Mono-Signals:", signal.shape)
        return signal

    def _get_audio_sample_path(self, index):
        # Definieren des Wurzelverzeichnisses directory
        directory = self.audio_dir
        # Extrahieren der Track-ID.wav → Excel kann keine vorangestellten Nullen für CSV generieren
        track_id = self.annotations.loc[index, 'WAV-name']
        # Formatieren der Track-ID als korrekten mit führenden Nullen -> jeder Name hat exakt 10 Chars '123456.wav'
        filename = str(track_id).zfill(10)
        # Pfad zum Audiofile erstellen
        path = os.path.join(directory, filename)
        return path

    def _get_audio_sample_label(self, index):
        genre_to_label = {'Blues': 0,
                          'Classical': 1,
                          'Country': 2,
                          'Easy Listening': 3,
                          'Electronic': 4,
                          'Experimental': 5,
                          'Folk': 6,
                          'Hip-Hop': 7,
                          'Instrumental': 8,
                          'International': 9,
                          'Jazz': 10,
                          'Old-Time / Historic': 11,
                          'Pop': 12,
                          'Rock': 13,
                          'Soul-RnB': 14,
                          'Spoken': 15}
        genre_name = self.annotations.iloc[index]["genre_top"]
        label = genre_to_label[genre_name]
        return label


"""
# Tests und Mel-Specs:
if __name__ == "__main__":
    ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
    AUDIO_DIR = 'C:/AI_Datasets/fma_medium/wav'
    SAMPLE_RATE = 44100
    NUM_SAMPLES = 1321967

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Benutze {device} zum berechne.")

    # Transformation/ Vorverarbeitung: Audio in Mel-Spektogramm wandeln
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,  # Hier die tatsächliche Abtastrate verwenden
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                    AUDIO_DIR,
                                    mel_spectrogram,
                                    SAMPLE_RATE, NUM_SAMPLES,
                                    device)

    # Plot des ersten Mel-Spektrogramms mit oben gewähltem Beispiel
    mel_spec, label = fmamed[30]
    print(f"Mel_Spec_Shape: {mel_spec.shape} Label:{label}")
    mel_spec = mel_spec.squeeze(0)  # Reduzieren der Kanaldimension
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.log2().detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.xlabel('Zeit in s')
    plt.ylabel(f'Mel filter <= {mel_spectrogram.n_mels}')
    plt.title(f'Mel Spektrogramm für Genre: {label}')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


    # Plot des ersten Mel-Spektrogramms mit oben gewähltem Beispiel
    mel_spec, label = fmamed[133]
    mel_spec = mel_spec.squeeze(0)  # Reduzieren der Kanaldimension
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.log2().detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.xlabel('Zeit in s')
    plt.ylabel(f'Mel filter <= {mel_spectrogram.n_mels}')
    plt.title(f'Mel Spektrogramm für Genre: {label}')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    

    # Überprüfen Sie den Typ und die Form des 'waveform'-Tensors
    for i in range(len(fmamed)):
        audio_sample_path, label = fmamed._get_audio_sample_path(i), fmamed._get_audio_sample_label(i)
        signal, sr = torchaudio.load(audio_sample_path)
        length_signal = signal.shape[1]
        # Right Padding testen
        if length_signal < NUM_SAMPLES:
            num_missing_samples = NUM_SAMPLES - length_signal
            last_dim_padding = (0, num_missing_samples)  # (left Padding Anz, right Padding Anz)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        # Cutting testen
        if signal.shape[1] > NUM_SAMPLES:
            signal = signal[:, :NUM_SAMPLES]
        print("Pfad:", audio_sample_path)
        print("Typ des Signals:", type(signal))
        print("Form des Signals:", signal.shape)
        mel_spec = mel_spectrogram(signal).to(device)  # Mel-Spektrogramm berechnen


    # Durch das Dataset iterieren und Pfade und Labels ausgeben
    # Überprüfe die Dimensionen des Inputs

    for i in range(len(fmamed)):
        audio_sample_path, label = fmamed._get_audio_sample_path(i), fmamed._get_audio_sample_label(i)
        print("Pfad:", audio_sample_path)
        print("Label:", label)
"""
