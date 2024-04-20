from collections import Counter

import torch
import random
import numpy as np
import torchaudio.transforms
import torchsampler
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, TensorDataset
import torchvision.models as models
from torchvision.models import VGG19_Weights, VGG19_BN_Weights
from torchaudio.models import RNNT, Conformer
import torchvision.transforms as transforms
# from FMA_Medium_ImageDataset import FreeMusicArchiveMedium
from FMA_Medium_AudioDataset import FreeMusicArchiveMedium
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from sklearn.preprocessing import OneHotEncoder

# Konstanten
BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 0.001
# L2-Regulierung / Norm-Penalisierung
WEIGHT_DECAY = 0.0001
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
AUDIO_DIR = "C:/AI_Datasets/fma_medium/wav/"
NUM_SAMPLES = 1321960
SAMPLE_RATE = 44100
cep_lifter = 50
N_MFCC = 13
N_FTT = 2048
HOP_LENGTH = 512
N_MELS = 64
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1
# vorbereitete Glob Vars für Auswertung:
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []


def split_data(dataset, train_percent=TRAIN_PERCENT, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT):
    # Berechne die Anzahl der Datenpunkte für jedes Split
    num_data = len(dataset)
    num_train = int(train_percent * num_data)
    num_val = int(val_percent * num_data)
    num_test = num_data - num_train - num_val
    print(f"Gesamt{num_data}, Train: {num_train}, Val{num_val}, Test{num_test} -> SUM {num_test + num_val + num_train}")
    # Verwende random_split, um die Daten automatisch aufzuteilen
    train_data, val_data, test_data = random_split(dataset, [num_train, num_val, num_test])

    return train_data, val_data, test_data


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # print(f"pre binarize y true:{y_true}")
    # print(f"pre binarize y pred:{y_true}")
    num_classes = len(np.unique(y_true))
    # print(f"klassen:{num_classes}")
    # Binarisieren der Labels
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    y_pred_binarized = label_binarize(y_pred, classes=range(num_classes))
    # print(f"bin y_pred{y_pred_binarized}")
    # print(f"bin y_true{y_true_binarized}")
    # Berechnen der Metriken
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

    # Berechnen der ROC-AUC. Es ist wichtig anzumerken, dass roc_auc_score multiklassen-AUC für Sie berechnet.
    auc_roc = roc_auc_score(y_true_binarized, y_pred_binarized, average='macro', multi_class='ovr')

    return accuracy, precision, recall, f1, auc_roc


def create_data_loader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    model.train()  # Setze Modell in den Trainingsmodus
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    y_true = []
    y_pred = []
    class_distribution = Counter()
    with tqdm(total=len(data_loader), desc="Epoch Training") as pbar:
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Berechnen der Vorhersagen und des Verlusts
            outputs = model(inputs)
            # print(f"before out{outputs}")
            loss = loss_fn(outputs, targets)
            # print(f"before actual{targets}")
            # Backpropagation und Optimierung
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Berechnen der Genauigkeit
            predicted = torch.argmax(outputs, dim=1)
            # print(f"\nafter predicted:\n{predicted}")
            # print(f"\nafter actual:\n{targets}")
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Verfolgen des Verlusts für die Ausgabe
            running_loss += loss.item() * inputs.size(0)

            # Verfolgen der Vorhersagen für Metriken
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # Aktualisieren der Klassenverteilungszähler
            class_distribution.update(targets.cpu().numpy())

            # Fortschrittsanzeige
            pbar.update(1)
            pbar.set_postfix({'loss': running_loss / total_samples})

        # Berechnen der Durchschnittsverlust und der Genauigkeit für die Epoche
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        # Werte für Auswertung erhalten:
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        # Berechnen der Metriken
        accuracy, precision, recall, f1, auc_roc = compute_metrics(y_true, y_pred)
        # Ausgabe der Klassenverteilung nach jeder Epoche
        print("Class distribution in the current epoch:", class_distribution)
        # Ausgabe von Verlust und Metriken
        print(f"\nLoss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {auc_roc:.4f}")

        return epoch_loss, epoch_accuracy


def calculate_class_weights(dataset):
    class_weights = {}
    class_counts = {}
    total_samples = len(dataset)

    # Zähle die Anzahl der Instanzen jeder Klasse
    for _, label in dataset:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Berechne die Gewichte entsprechend der Klassenanzahl
    for label, count in class_counts.items():
        weight = total_samples / (count * len(class_counts))
        class_weights[label] = weight
        print(f"Klasse: {label} erhält Gewichtung: {weight}")
    weight_list = [class_weights[label] for label in sorted(class_weights.keys())]
    return weight_list


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
                # Test für Vergleichbarkeit bei Berechnung der loss_fn
                # print(f"Label: {targets}")
                # Test für Vergleichbarkeit bei Berechnung der loss_fn

                loss = loss_fn(outputs, targets)

                # Berechne die Genauigkeit
                predicted = torch.argmax(outputs, dim=1)
                # Test für Vergleichbarkeit bei Berechnung der loss_fn print(f"Vorhersage: {predicted}")
                # print(f"Vorhersage: {predicted}")
                correct_predictions += (predicted == targets).sum().item()
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
            # Werte für Auswertung erhalten:
            val_losses.append(epoch_loss)
            val_accuracies.append(epoch_accuracy)

            # Berechne die Metriken
            accuracy, precision, recall, f1, auc_roc = compute_metrics(y_true, y_pred)

            # Gib den Verlust und die Metriken aus
            print(f"\nValidation: \nvLoss: {epoch_loss:.4f}, vAccuracy: {epoch_accuracy:.4f}, "
                  f"vPrecision: {precision:.4f}, vRecall: {recall:.4f}, "
                  f"vF1-Score: {f1:.4f}, vROC-AUC: {auc_roc:.4f}")

            return epoch_loss, epoch_accuracy


def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_accuracy = train_single_epoch(model, train_data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        val_loss, val_accuracy = validate(model, val_data_loader, loss_fn, device)
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
    print(f"Lade Datensatzklasse FMAMedium")

    # MFCCs
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FTT,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "mel_scale": "htk",
        },
    )

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                    AUDIO_DIR,
                                    mfcc,
                                    SAMPLE_RATE,
                                    NUM_SAMPLES,
                                    device)
    print(f"{fmamed}")
    # Verwende die Funktion split_data, um die Daten aufzuteilen
    print("Erstelle Trainings-, Test- und Validierungsdaten...")
    train_data, val_data, test_data = split_data(fmamed)


    def _subset_to_tensordataset(subset):
        subset_data = [sample[0] for sample in subset]
        subset_labels = [sample[1] for sample in subset]

        # Konvertiere die Daten und Labels in Tensoren
        data_tensor = torch.stack(subset_data)
        labels_tensor = torch.tensor(subset_labels)
        # Erstelle ein TensorDataset aus den Tensoren
        tensor_dataset = TensorDataset(data_tensor, labels_tensor)
        return tensor_dataset

    print("Wandle TrainingsSubset in Tensor Dataset um:")
    train_tensor = _subset_to_tensordataset(train_data)
    # Zähle die Anzahl der Samples pro Klasse vor dem Sampling
    class_counts_before = Counter([sample[1] for sample in train_data])
    # Erstellen eines ImbalancedDatasetSampler mit den berechneten Gewichten
    sampler = torchsampler.ImbalancedDatasetSampler(train_tensor)

    # Erstelle Daten-Loader für Trainings-, Validierungs- und Testdaten
    print("Dataloader Trainingsdaten.")
    # Erstelle den DataLoader mit dem Sampler
    train_dataloader = DataLoader(train_tensor, batch_size=BATCH_SIZE, sampler=sampler)
    # Zähle die Anzahl der Samples pro Klasse nach dem Sampling
    print("Klassenverteilung vor dem Sampling:", class_counts_before)
    print("Dataloader Validierungsdaten.")
    val_dataloader = create_data_loader(_subset_to_tensordataset(val_data), batch_size=BATCH_SIZE)
    print("Dataloader Testdaten.")
    test_dataloader = create_data_loader(_subset_to_tensordataset(test_data), batch_size=BATCH_SIZE)

    # Nutzen des vortraineirten Pytorch VGG19
    print("RNNT erstellen.")


    class Transcriber(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(Transcriber, self).__init__()
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        def forward(self, x):
            out, _ = self.rnn(x)
            return out


    class Predictor(nn.Module):
        def __init__(self, input_size, output_size):
            super(Predictor, self).__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, x):
            out = self.fc(x)
            return out


    class Joiner(nn.Module):
        def __init__(self, transcriber_output_size, predictor_output_size, hidden_size, num_classes):
            super(Joiner, self).__init__()
            self.fc1 = nn.Linear(transcriber_output_size + predictor_output_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, transcriber_output, predictor_output):
            combined_output = torch.cat((transcriber_output, predictor_output), dim=1)
            out = self.fc1(combined_output)
            out = self.relu(out)
            out = self.fc2(out)
            return out


    transcriber = Transcriber(input_size=N_MFCC, hidden_size=64, num_layers=2)  # 2 LSTM Layer
    predictor = Predictor(input_size=64, output_size=11)  # input_size=hidden_size des transcribers output=11 Klassen
    joiner = Joiner(transcriber_output_size=64, predictor_output_size=11, hidden_size=128, num_classes=11)
    RNNT = torchaudio.models.RNNT(transcriber, predictor, Joiner)

    """
    # Die Klassen sind nicht balanciert, daher:
    class_weights = calculate_class_weights(train_dataloader.dataset)
    
    # initialisiere loss function + optimiser
    class_weights = calculate_class_weights(train_dataloader.dataset)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    """
    loss_fn = torch.nn.CTCLoss()
    # Weight Decay als L2-Regulierung als Maßnahme gegen Overfitting
    optimiser = torch.optim.Adam(RNNT.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # train model
    model = RNNT.to(device)
    train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Erstellen der Diagramme
    epochs = range(1, EPOCHS + 1)

    # Trainings- und Validierungsverluste
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train vs Validation Loss')
    plt.legend()
    plt.show()

    # Trainings- und Validierungsgenauigkeit
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.legend()
    plt.show()
    # Testen Sie das Modell auf den Testdaten
    print("Testen des Modells...\n")
    test_loss, test_accuracy = validate(model, test_dataloader, loss_fn, device)

    # Ausgabe der Ergebnisse
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # save model
    torch.save(model.state_dict(), "VGG19_fma_med.pth")
    print("Trainiertes Netz als cnn_fma_med.pth gespeichert.")
