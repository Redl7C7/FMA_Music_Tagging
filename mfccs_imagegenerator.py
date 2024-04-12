import os
import torchaudio
import torchaudio.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

cep_lifter = 22

if __name__ == "__main__":
    # Verzeichnis, in das die Bilder gespeichert werden sollen
    wav_directory = "C:/AI_Datasets/fma_medium/wav"
    image_directory = 'C:/AI_Datasets/fma_medium/mfcc-images'
    os.makedirs(image_directory, exist_ok=True)

    # Pfad zur Annotationsdatei
    annotations_file = 'C:/AI_Datasets/Tracks_Medium.csv'

    # Lade Annotationsdatei
    annotations = pd.read_csv(annotations_file, delimiter=';')  # Semikolon als Trennzeichen anpassen

    # Transformation für MFCCs
    mfcc_transform = transforms.MFCC(sample_rate=44100, n_mfcc=13)
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
        cep_lifter= 22
        # Überprüfen, ob das Bild bereits existiert
        if os.path.exists(image_path):
            print(f"Bild {image_path} existiert bereits, überspringe Konvertierung.")
            continue


        # Lade die Wave-Datei
        waveform, sample_rate = torchaudio.load(wav_path)
        print(f"{waveform.shape}")
        # Überprüfen, ob die Waveform mono ist
        if waveform.shape[0] > 1:
            # Falls die Waveform mehr als einen Kanal hat, wähle den ersten Kanal aus
            waveform = waveform[0]
        print(f"{waveform.shape}")
        # Erzeuge die MFCCs und füge eine Batch-Dimension hinzu
        mfcc = mfcc_transform(waveform).squeeze().detach().numpy()
        print(f"{mfcc.shape}")
        # Sinusförmiges Anheben der MFCCs
        nframes, ncoeff = mfcc.shape
        n = np.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfcc *= lift

        # Konvertiere MFCCs in ein Bildformat (z.B. als Graustufenbild)
        mfcc_image = np.uint8((mfcc - mfcc.min()) / (mfcc.max() - mfcc.min()) * 255)

        # Erzeuge ein PIL-Image-Objekt
        mfcc_image = Image.fromarray(mfcc_image)

        # Speichere das Bild
        mfcc_image.save(image_path)

        print(f"Bild {filename[:-4]}.png gespeichert unter {image_path}")