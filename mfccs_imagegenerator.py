import os
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

cep_lifter = 25
N_MFCC = 13
N_FTT = 4096
HOP_LENGTH = 512
N_MELS = 256
if __name__ == "__main__":
    # Verzeichnis, in das die Bilder gespeichert werden sollen
    wav_directory = "C:/AI_Datasets/fma_medium/wav"
    image_directory = 'C:/AI_Datasets/fma_medium/bunt-mfcc-images'
    os.makedirs(image_directory, exist_ok=True)

    # Pfad zur Annotationsdatei
    annotations_file = 'C:/AI_Datasets/Tracks_Medium.csv'

    # Lade Annotationsdatei
    annotations = pd.read_csv(annotations_file, delimiter=';')  # Semikolon als Trennzeichen anpassen

    # Transformation für MFCCs
    mfcc_transform = transforms.MFCC(sample_rate=44100, n_mfcc=N_MFCC, melkwargs={
        "n_fft": N_FTT,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "mel_scale": "htk",
    },)
    # Schleife über alle Zeilen in der CSV
    for index, row in annotations.iterrows():
        # Extrahiere den Dateinamen aus der entsprechenden Spalte der Zeile
        filenum = row['WAV-name']  # Hier musst du den Namen der Spalte anpassen, die den Dateinamen enthält
        filename = str(filenum).zfill(10)
        segment_folder = str(filename)[:3]
        # Konstruieren des vollständigen Pfads zur Wave-Datei
        wav_path = os.path.join(wav_directory, filename[:-4] + '.wav')
        # Konstruieren des vollständigen Pfads zum Zielbild
        image_path = os.path.join(image_directory, filename[:-4] + '.png')
        # Überprüfen, ob das Bild bereits existiert
        if os.path.exists(image_path):
            print(f"Bild {image_path} existiert bereits, überspringe Konvertierung.")
            continue


        # Lade die Wave-Datei
        waveform, sample_rate = torchaudio.load(wav_path)
        # Überprüfen, ob die Waveform mono ist
        if waveform.shape[0] > 1:
            # Falls die Waveform mehr als einen Kanal hat, wähle den ersten Kanal aus
            waveform = waveform[0]
        # Erzeuge die MFCCs und füge eine Batch-Dimension hinzu
        mfcc = mfcc_transform(waveform).squeeze().detach().numpy()
        # Sinusförmiges Anheben der MFCCs
        nframes, ncoeff = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift
        """
        # Plotte das MFCC-Diagramm
        duration = waveform.size(1) / sample_rate
        plt.figure(figsize=(10, 4))
        # plt.imshow(mfcc.T, origin='lower', aspect='auto', cmap='viridis')
        plt.imshow(mfcc, origin='lower', aspect='auto', cmap='viridis',
                   extent=[0, duration, 0, N_MFCC])
        plt.colorbar(label='Amplitude')
        plt.xlabel('Zeit')
        plt.ylabel('MFCC Koeffizienten')
        plt.title(f'MFCC-Diagramm von {filename}')
        plt.show()
        """
        # plt.figure(figsize=(10, 5))
        plt.figure(figsize=(2.9, 2.92)) # auf meinem PC erzeugt es 224x224 PNG Files
        duration = waveform.size(1) / sample_rate
        # plt.figure(figsize=(224/100, 224/100))
        # rotate
        mfcc = np.rot90(mfcc, k=1)
        # eigene Colormap für bessere Kontrraste
        # Definieren der Farben für die Colormap + Verhältnis zwischen Position und Farbe
        colors = [(0, 'black'),
                  (0.1, 'purple'),
                  (0.2, 'blue'),
                  (0.4, 'green'),
                  (0.5, 'red'),
                  (0.6, 'yellow'),
                  (0.8, 'orange'),
                  (0.9, 'magenta'),
                  (1, 'white')]
        # Erstellen der Colormap
        custom_cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        plt.imshow(mfcc, origin='lower', aspect='auto', cmap=custom_cmap, extent=[0, duration, 0, N_MFCC])
        plt.axis('off')
        # plt.savefig(image_path)  # Speichern des Diagramms als PNG
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

        print(f"Bild {filename[:-4]}.png gespeichert unter {image_path}")
