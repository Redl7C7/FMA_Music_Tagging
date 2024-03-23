import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio

directory = r'C:/AI_Datasets/fma_medium/fma_medium'
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        if filename.endswith('.mp3'):
            path = os.path.join(dirpath, filename)
            print(path)


class FMADataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        track_id = self.data.iloc[idx, 1]
        track_path = self.find_track_path(track_id)
        return track_path

    def find_track_path(self, track_id):
        track_path = os.path.join(self.root_dir, 'tracks', f'{track_id}.mp3')
        if os.path.exists(track_path):
            return track_path
        return None


# Beispielaufruf
csv_file = 'C:/AI_Datasets/Tracks_Medium.csv'
root_dir = 'C:/AI_Datasets/fma_medium/fma_medium'
fma_dataset = FMADataset(csv_file, root_dir)
data_loader = DataLoader(fma_dataset, batch_size=32, shuffle=True)

for track_paths in data_loader:
    # Hier kannst du die Track-Pfade für jeden Batch von Daten verwenden
    for path in track_paths:
        print(path)
