import os
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Verzeichnis, in das die Bilder gespeichert werden sollen
wav_directory = "C:/AI_Datasets/fma_medium/wav"
image_directory = 'C:/AI_Datasets/fma_medium/mel-spec-images'
os.makedirs(image_directory, exist_ok=True)

# Pfad zur Annotationsdatei
annotations_file = 'C:/AI_Datasets/Tracks_Medium.csv'

# Lade Annotationsdatei
annotations = pd.read_csv(annotations_file, delimiter=';')  # Semikolon als Trennzeichen anpassen

# Transformation für Mel-Spektrogramme
mel_spec_transform = transforms.MelSpectrogram(sample_rate=44100, n_fft=1024, hop_length=512, n_mels=64)

# Schleife über alle Zeilen in der CSV
for index, row in annotations.iterrows():
    # Extrahiere den Dateinamen aus der entsprechenden Spalte der Zeile
    filenum = row['Filename']  # Hier musst du den Namen der Spalte anpassen, die den Dateinamen enthält
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

    try:
        # Lade die Wave-Datei
        waveform, sample_rate = torchaudio.load(wav_path)

        # Überprüfen, ob die Waveform mono ist
        if waveform.shape[0] > 1:
            # Falls die Waveform mehr als einen Kanal hat, wähle den ersten Kanal aus
            waveform = waveform[0]

        # Erzeuge das Mel-Spektrogramm und füge eine Batch-Dimension hinzu
        mel_spec = mel_spec_transform(waveform.unsqueeze(0)).squeeze(0).detach().numpy()

        # Logarithmische Skalierung
        mel_spec = np.log1p(mel_spec)

        # Auswahl des ersten Kanals
        mel_spec = mel_spec[0]

        # Datentypkonvertierung
        mel_spec = mel_spec.astype(np.float32)

        # Plotte das Mel-Spektrogramm
        plt.figure(figsize=(5, 5))
        plt.imshow(mel_spec, cmap='viridis', origin='lower', aspect='auto')
        plt.axis('off')

        # Speichere das Bild mit 224x224 Pixeln
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=224 / 5)  # Hier dpi entsprechend anpassen

        # Öffne das gespeicherte Bild mit Pillow
        img = Image.open(image_path)

        # Skaliere das Bild auf 224x224 Pixel
        img = img.resize((224, 224), Image.BILINEAR)  # Verwende BILINEAR-Interpolation

        # Speichere das skalierte Bild
        img.save(image_path)

        plt.close()  # Schließe das Plotfenster

        print(f"Bild {filename[:-4]}.png gespeichert unter {image_path}")
    except Exception as e:
        print(f"Fehler beim Laden von {wav_path}: {e}")
