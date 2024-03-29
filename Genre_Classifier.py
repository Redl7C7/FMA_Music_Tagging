import torch
from torch import nn
from torch.utils.data import DataLoader
from FMA_Medium_Data import FreeMusicArchiveMedium
from CNN_FMA_Med import CNNetwork
import torchaudio

# Konstanten
BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 0.1
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
AUDIO_DIR = "C:/AI_Datasets/fma_medium/wav"
NUM_SAMPLES = 1321967
SAMPLE_RATE = 22050


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiate dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    print(f"Lade Datasetklasse FMAMedium")
    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                    AUDIO_DIR,
                                    mel_spectrogram,
                                    SAMPLE_RATE,
                                    NUM_SAMPLES,
                                    device)

    train_dataloader = create_data_loader(fmamed, BATCH_SIZE)

    # Modell erzeugen und CUDA zuordnen
    cnn = CNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn_fma_med.pth")
    print("Trained feed forward net saved at cnn_fma_med.pth")