import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd


class FreeMusicArchiveMedium(Dataset):
    def __init__(self, annotations_file, image_dir, transformation, device):
        self.annotations = pd.read_csv(annotations_file, delimiter=';')
        self.image_dir = image_dir
        self.transformation = transformation
        self.device = device
        # Neue Zuordnung von Genres zu Labels
        self.genre_to_label = {'Classical': 0,
                               'Electronic': 1,
                               'Experimental': 2,
                               'Folk': 3,
                               'Hip-Hop': 4,
                               'Instrumental': 5,
                               'International': 6,
                               'Jazz': 7,
                               'Old-Time / Historic': 8,
                               'Pop': 9,
                               'Rock': 10,
                               'Other': 11}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_num = self.annotations.iloc[index, 0]
        img_name = str(img_num).zfill(6)
        img_name = img_name + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        # Öffnen und Konvertieren des Bildes in das richtige Format (RGBA --> RGB)
        image = Image.open(img_path).convert('RGB')
        label = self._get_image_label(index)
        transformed_image = self.transformation(image)
        image.close()  # Schließe das Image-Objekt, um die Datei freizugeben
        return transformed_image, label

    def _get_image_label(self, index):
        genre_name = self.annotations.iloc[index]["genre_top"]

        # Überprüfen, ob das Genre in der Liste der zusammenzufassenden Genres ist
        if genre_name in ['Easy Listening', 'Blues', 'Spoken', 'Soul-RnB', 'Country']:
            label = self.genre_to_label['Other']
        else:
            label = self.genre_to_label[genre_name]
        return label
