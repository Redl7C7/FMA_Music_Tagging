import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torchaudio.utils import download_asset

print(torch.__version__)
print(torchaudio.__version__)

import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display as IPD import Audio

#Seeds sind Randomizer für numerische Werte, wird benötigt um Daten durchzumischen oder initiale Gewqichtungen zu vergeben
torch.random.manual_seed(0)

#Multiskalenanalyse des Audiosignals/ Musikstücks
#Multiskalenanalyse ist ein Spektogramm mit verschiedener Intervallgröße oder auch N für die Fast Fourier Transformation (FFT)
#Es wird daher drei Auflösungen für diese Arbeit geben N=512, N=1024 und N=2048

