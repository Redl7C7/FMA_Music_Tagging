import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class FreeMusicArchiveMedium(Dataset):
    def __init__(self, annotations_file, image_dir, device):
        self.annotations = pd.read_csv(annotations_file, delimiter=';')
        self.image_dir = image_dir
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_num  = self.annotations.iloc[index, 'track_id']
        img_name = str(img_num).zfill(6)
        img_name = img_name + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        label = self._get_image_label(index)
        return image, label

    def _get_image_label(self, index):
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
