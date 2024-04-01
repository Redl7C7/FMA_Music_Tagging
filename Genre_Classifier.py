import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import VGG19_Weights

from FMA_Medium_Data import FreeMusicArchiveMedium
from CNN_FMA_Med import CNNetwork
from FMA_med_VGG_Modules import VGG, VGG_types
import torchaudio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score

# Konstanten
BATCH_SIZE = 24
EPOCHS = 2
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
AUDIO_DIR = "C:/AI_Datasets/fma_medium/wav"
NUM_SAMPLES = 1321967
SAMPLE_RATE = 44100


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()  # Setze Modell in den Trainingsmodus
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Berechnen der Vorhersagen und des Verlusts
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backpropagation und Optimierung
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Berechnen der Genauigkeit
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)

        # Verfolgen des Verlusts für die Ausgabe
        running_loss += loss.item() * inputs.size(0)

        # Verfolgen der Vorhersagen für Metriken
        y_true.extend(targets.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    # Berechnen der Durchschnittsverlust und der Genauigkeit für die Epoche
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_accuracy = correct_predictions / total_samples

    # Berechnen der Metriken
    accuracy, precision, recall, f1, roc_auc, pr_auc = compute_metrics(y_true, y_pred)

    # Ausgabe von Verlust und Metriken
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, "
          f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

    return epoch_loss, epoch_accuracy


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_accuracy = train_single_epoch(model, data_loader, loss_fn, optimiser, device)
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

    # Nutzen des vorgestalteten Pytorch VGG19
    VGG19 = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device)
    # Die Eingabeschicht des VGG19-Modells ändern, um mit den Spektrogramm-Eingabedaten umzugehen
    VGG19.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(VGG19.parameters(), lr=LEARNING_RATE)
    # train model
    train(VGG19, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(VGG19.state_dict(), "VGG19_fma_med.pth")
    print("Trainiertes Netz als cnn_fma_med.pth gespeichert.")

    # Modell erzeugen und CUDA zuordnen
    """
    # cnn = CNNetwork().to(device)
    VGG19 = VGG(
        in_channels=1,
        in_height=224,
        in_width=224,
        architecture=VGG_types["VGG19"]
    ).to(device)
    print(VGG19)
    """
