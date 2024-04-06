import torch
import random
import numpy as np
from torch import nn
from torchvision.models import VGG19_Weights
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from FMA_Medium_MelSpecs import FreeMusicArchiveMedium
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
import torch.nn.functional as F

# Konstanten
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
IMAGE_DIR = "C:/AI_Datasets/fma_medium/mel-spec-images"


def split_data(dataset, train_percent=0.5, val_percent=0.25, test_percent=0.25):
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


def compute_metrics(y_true, y_pred, device):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Falls y_pred eine 1D-Array ist, wird sie in eine Spalte einer 2D-Array umgewandelt
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    probabilities = F.softmax(torch.tensor(y_pred).to(device), dim=1)  # Anwendung der Softmax-Funktion auf die Ausgabe

    accuracy = accuracy_score(y_true, np.argmax(probabilities.cpu().detach().numpy(), axis=1))
    precision = precision_score(y_true, np.argmax(probabilities.cpu().detach().numpy(), axis=1), average='macro', zero_division=1)
    recall = recall_score(y_true, np.argmax(probabilities.cpu().detach().numpy(), axis=1), average='macro', zero_division=1)
    f1 = f1_score(y_true, np.argmax(probabilities.cpu().detach().numpy(), axis=1), average='macro')
    pr_auc = average_precision_score(y_true, np.argmax(probabilities.cpu().detach().numpy(), axis=1), average='macro')

    # Berechne ROC-AUC für Multi-Klassen Klassifikation
    roc_auc = roc_auc_score(y_true, probabilities.cpu().detach().numpy(), average='macro', multi_class='ovr')

    return accuracy, precision, recall, f1, pr_auc, roc_auc


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
    with tqdm(total=len(data_loader), desc="Epoch Training") as pbar:
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Berechnen der Vorhersagen und des Verlusts
            # Berechnen der Vorhersagen und des Verlusts
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            # Ausgabe der Labels und Vorhersagen
            print("Labels:", targets)
            print("Vorhersagen:", outputs)


            # Backpropagation und Optimierung
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Berechne die Genauigkeit
            _, predicted = torch.max(outputs, 1)
            correct_predictions += torch.sum((predicted == targets).int()).item()
            total_samples += targets.size(0)

            # Verfolgen des Verlusts für die Ausgabe
            running_loss += loss.item() * inputs.size(0)

            # Verfolgen der Vorhersagen für Metriken
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Fortschrittsanzeige
            pbar.update(1)
            pbar.set_postfix({'loss': running_loss / total_samples})

        # Berechnen der Durchschnittsverlust und der Genauigkeit für die Epoche
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples

        # Berechnen der Metriken
        accuracy, precision, recall, f1, pr_auc, roc_auc = compute_metrics(y_true, y_pred, device)

        # Ausgabe von Verlust und Metriken
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, "
              f"PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

        return epoch_loss, epoch_accuracy, accuracy, precision, recall, f1, pr_auc


def validate(model, data_loader, loss_fn, device):
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Validation") as pbar:
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Berechne die Vorhersagen und den Verlust
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Berechne die Genauigkeit
                _, predicted = torch.max(outputs, 1)
                correct_predictions += torch.sum((predicted == targets).int()).item()
                total_samples += targets.size(0)

                # Verfolge den Verlust für die Ausgabe
                running_loss += loss.item() * inputs.size(0)

                # Verfolge die Vorhersagen für Metriken
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                # Fortschrittsanzeige
                pbar.update(1)
                pbar.set_postfix({'loss': running_loss / total_samples})

            # Berechne den Durchschnittsverlust und die Genauigkeit für die Validierung
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_accuracy = correct_predictions / total_samples

            # Berechne die Metriken
            accuracy, precision, recall, f1, pr_auc, roc_auc = compute_metrics(y_true, y_pred, device)

            return epoch_loss, epoch_accuracy, accuracy, precision, recall, f1, pr_auc


def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_accuracy, _, _, _, _, _ = train_single_epoch(model, train_data_loader, loss_fn, optimiser,
                                                                       device)
        print("---------------------------")
        v_loss, v_accuracy, v_precision, v_recall, v_f1, v_pr_auc, v_roc_auc = validate(model, val_data_loader, loss_fn,
                                                                                        device)
        print(f"\nValidation Loss: {v_loss:.4f}, Validation Accuracy: {v_accuracy:.4f}, "
              f"Validation Precision: {v_precision:.4f}, Validation Recall: {v_recall:.4f}, "
              f"Validation F1-Score: {v_f1:.4f}, Validation PR-AUC: {v_pr_auc:.4f}, "
              f"Validierung ROC-AUC: {v_roc_auc:.4f}")
        print("---------------------------")
    print("Training beendet.")


def test(model, test_data_loader, loss_fn, device):
    model.eval()  # Setze das Modell in den Evaluierungsmodus
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        with tqdm(total=len(test_data_loader), desc="Testing") as pbar:
            for inputs, targets in test_data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Berechne die Vorhersagen und den Verlust
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Berechne die Genauigkeit
                _, predicted = torch.max(outputs, 1)
                correct_predictions += torch.sum((predicted == targets).int()).item()
                total_samples += targets.size(0)

                # Verfolge den Verlust für die Ausgabe
                running_loss += loss.item() * inputs.size(0)

                # Verfolge die Vorhersagen für Metriken
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

                # Fortschrittsanzeige
                pbar.update(1)
                pbar.set_postfix({'loss': running_loss / total_samples})

            # Berechne den Durchschnittsverlust und die Genauigkeit für den Test
            test_loss = running_loss / len(test_data_loader.dataset)
            test_accuracy = correct_predictions / total_samples

            # Berechne die Metriken
            accuracy, precision, recall, f1, pr_auc, roc_auc = compute_metrics(y_true, y_pred, device)

            # Gib den Verlust und die Metriken aus
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, "
                  f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, "
                  f"Test F1-Score: {f1:.4f}, Test PR-AUC: {pr_auc:.4f}")

            return test_loss, test_accuracy


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
    # test_dataloader = create_data_loader(test_data, batch_size=BATCH_SIZE)

    # Nutzen des vorgestalteten Pytorch VGG19
    print("vgg19 erstellen.")
    VGG19 = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device)
    # VGG19 = models.vgg19(weights=None).to(device)
    # Die Eingabeschicht des VGG19-Modells ändern, um mit den Spektrogramm-Eingabedaten umzugehen
    print("Eingang des VGG19 auf Spektogramme in Tensor anpassen.")
    # VGG19.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)
    # initialisiere loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(VGG19.parameters(), lr=LEARNING_RATE)
    # train model
    train(VGG19, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)
    # save model
    torch.save(VGG19.state_dict(), "VGG19_fma_med.pth")
    print("Trainiertes Netz als cnn_fma_med.pth gespeichert.")
    print("Lade Testdaten.")
    test_dataloader = create_data_loader(test_data, batch_size=BATCH_SIZE)
    # Testen des Modells
    test_loss, test_accuracy = test(VGG19, test_dataloader, loss_fn, device)
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
