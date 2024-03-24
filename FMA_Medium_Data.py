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
        self.transformation = transformation.to(self.device)
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
        signal, sr = torchaudio.load(audio_sample_path, format="MP3")
        # signal, sr = torchaudio.load()
        if signal is None:
            print("Fehler beim Laden der Audiodatei.")
            # Füge hier weitere Fehlerbehandlung hinzu, falls erforderlich
        else:
            print("Audiodaten erfolgreich geladen.")

        # Überprüfe die Abtastrate
        print("Abtastrate (sr):", sr)

        # Überprüfe die Form der Audiodaten
        print("Form der Audiodaten (Signal):", signal.shape)
        signal = signal.to(self.device)
        # Normalisierungen
        # gleiche Sample-RATE
        # signal = self._resample_if_necessary(signal, sr)
        # eindimensionale Eingabe (1 Kanal)
        signal = self._mix_down_if_necessary(signal)
        # Cut, wenn Song zu lang
        signal = self._cut_if_necessary(signal)
        # Zero Right Padding für kürzere Songs
        # signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        # Shape [1] ist beim Tensor-Tupel die Nummer der Samples
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
            return signal

    def _right_pad_if_necessary(self, signal):
        # Check if the signal is None (i.e., audio loading failed)
        if signal is None:
            # Return None if the signal is None
            return None

        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)  # (left Padding Anz, right Padding Anz)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] == 2:
            signal = torch.mean(signal, dim=0, keepdim=True)
        print("Form des Mono-Signals:", signal.shape)
        return signal

    def _get_audio_sample_path(self, index):
        # Definieren des Wurzelverzeichnisses directory
        directory = self.audio_dir
        # Extrahieren der Track-ID
        track_id = self.annotations.loc[index, 'track_id']
        # Formatieren der Track-ID mit führenden Nullen
        filename = str(track_id).zfill(6)
        # Extrahieren des Segment-Ordners, die ersten drei Zeichen des Filenames = Name des Unterverzeichnisses
        segment_folder = str(filename)[:3]
        # Pfad zum Audiofile erstellen
        path = os.path.join(directory, segment_folder, filename + '.mp3')
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index]["genre_top"]


"""
if __name__ == "__main__":
    ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
    AUDIO_DIR = 'C:/AI_Datasets/fma_medium/fma_medium'
    SAMPLE_RATE = 44100

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Benutze {device} zum berechne.")

    # Transformation/ Vorverarbeitung: Audio in Mel-Spektogramm wandeln
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                    AUDIO_DIR,
                                    mel_spectrogram,
                                    SAMPLE_RATE, Genre_Classifier.NUM_SAMPLES,
                                    device)

    # Für Versuche:
    # Beispiel mit Index "2" wählen
    # signal, sr = fmamed[2]

    
    # Plot des ersten Mel-Spektrogramms mit oben gewähltem Beispiel
    mel_spec, label = fmamed[12545]
    mel_spec = mel_spec.squeeze(0)  # Reduzieren der Kanaldimension
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec.log2().detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.xlabel('Zeit in s')
    plt.ylabel(f'Mel filter <= {mel_spectrogram.n_mels}')
    plt.title(f'Mel Spektrogramm für Genre: {label}')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    

    # Durch das Dataset iterieren und Pfade und Labels ausgeben
    # Überprüfe die Dimensionen des Inputs
    print(f"Dimensionen des Input-Mel-Spektrogramms: {mel_spectrogram(torch.randn(1, Genre_Classifier.NUM_SAMPLES).to(device)).shape}")

    for i in range(len(fmamed)):
        audio_sample_path, label = fmamed._get_audio_sample_path(i), fmamed._get_audio_sample_label(i)
        print("Pfad:", audio_sample_path)
        print("Label:", label)

"""
