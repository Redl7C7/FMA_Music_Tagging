import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from FMA_Medium_MelSpecs import FreeMusicArchiveMedium

# Konstanten
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
IMAGE_DIR = "C:/AI_Datasets/fma_medium/mel-spec-images"


def split_data(dataset, train_percent=0.01, val_percent=0.025, test_percent=0.025):
    # Berechne die Anzahl der Samples im Datensatz
    num_sample_data = len(dataset)
    num_train = int(train_percent * num_sample_data)
    num_val = int(val_percent * num_sample_data)
    num_test = num_sample_data - num_train - num_val

    # Erzeuge zufällige Indizes für den gesamten Datensatz
    indices = list(range(num_sample_data))
    random.shuffle(indices)

    # Teile die Indizes für Trainings-, Validierungs- und Testdaten auf
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # Erstelle Datenuntergruppen basierend auf den Indizes
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]

    return train_data, val_data, test_data


def create_data_loader(data, batch_size):
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()  # Setze Modell in den Trainingsmodus
    running_loss = 0.0
    total_samples = 0
    with tqdm(total=len(data_loader), desc="Epoch Training") as pbar:
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

            # Verfolgen des Verlusts für die Ausgabe
            running_loss += loss.item() * inputs.size(0)

            # Fortschrittsanzeige
            pbar.update(1)
            pbar.set_postfix({'loss': running_loss / total_samples})

    # Berechnen des Durchschnittsverlusts für die Epoche
    epoch_loss = running_loss / len(data_loader.dataset)

    return epoch_loss


def validate(model, data_loader, loss_fn, device):
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Validation") as pbar:
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Berechnen der Vorhersagen und des Verlusts
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Verfolgen des Verlusts für die Ausgabe
                running_loss += loss.item() * inputs.size(0)

                # Fortschrittsanzeige
                pbar.update(1)
                pbar.set_postfix({'loss': running_loss / total_samples})

        # Berechnen des Durchschnittsverlusts für die Validierung
        epoch_loss = running_loss / len(data_loader.dataset)

        return epoch_loss


def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_single_epoch(model, train_data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        v_loss = validate(model, val_data_loader, loss_fn, device)
        print(f"\nValidation Loss: {v_loss:.4f}")
        print("---------------------------")
    print("Training beendet.")


if __name__ == "__main__":
    # Geräte zum Training setzen
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # Datensatzklasse instanziieren
    print(f"Starte mit folgenden Params:"
          f"BATCH_SIZE = {BATCH_SIZE}",
          f"EPOCHS = {EPOCHS}",
          f"LEARNING_RATE = {LEARNING_RATE}",
          f"ANNOTATIONS_FILE = {ANNOTATIONS_FILE}",
          f"IMAGE_DIR = {IMAGE_DIR}")
    print(f"Lade Datensatzklasse FMAMedium")

    # Definiere die Transformationen
    transformation = transforms.Compose([
        transforms.ToTensor(),  # Wandle das Bild in einen Tensor um
    ])

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                    IMAGE_DIR,
                                    transformation,
                                    device)

    # Verwende die Funktion split_data, um die Daten aufzuteilen
    print("Erstelle Trainings-, Test- und Validierungsdaten...")
    train_data, val_data, test_data = split_data(fmamed)

    # Erstelle Daten-Loader für Trainings-, Validierungs- und Testdaten
    print("Trainingsdaten laden.")
    train_dataloader = create_data_loader(train_data, batch_size=BATCH_SIZE)
    print("Lade Validierungsdaten.")
    val_dataloader = create_data_loader(val_data, batch_size=BATCH_SIZE)

    # Nutzen des vorgestalteten Pytorch VGG19
    print("vgg19 erstellen.")
    VGG19 = models.vgg19(pretrained=True)
    num_classes = 16
    VGG19.classifier[6] = nn.Linear(4096, num_classes)
    VGG19.to(device)
    # Ausgabe des angepassten VGG19-Modells
    print(VGG19)

    # Eingang des VGG19-Modells anpassen, um mit den Spektrogramm-Eingabedaten umzugehen
    print("Eingang des VGG19 auf Spektogramme in Tensor anpassen.")

    # Initialisiere Loss-Funktion + Optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(VGG19.parameters(), lr=LEARNING_RATE)

    # Trainiere das Modell
    train(VGG19, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)
    #esdede
    # Speichere das trainierte Modell
    torch.save(VGG19.state_dict(), "VGG19_fma_med.pth")
    print("Trainiertes Netz als cnn_fma_med.pth gespeichert.")
