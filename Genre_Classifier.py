import torch
import random
import numpy as np
from sklearn.preprocessing import label_binarize
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from FMA_Medium_MelSpecs import FreeMusicArchiveMedium
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from sklearn.preprocessing import OneHotEncoder

# Konstanten
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.01
# L2-Regulierung / Norm-Penalisierung
WEIGHT_DECAY = 0.001
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
IMAGE_DIR = "C:/AI_Datasets/fma_medium/mel-spec-images"


# Erstelle dynamische Splits zur Laufzeit:
def split_data(dataset, train_percent=0.002, val_percent=0.25, test_percent=0.25):
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


def compute_metrics(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Berechnen der Metriken
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # Da Ihre Ausgabe keine Wahrscheinlichkeiten sind, sondern nur Vorhersagen, ist pr_auc nicht sinnvoll.
        pr_auc = None

        # Berechnen der ROC-AUC. Es ist wichtig anzumerken, dass roc_auc_score multiklassen-AUC für Sie berechnet.
        auc_roc = roc_auc_score(y_true, y_pred, average='macro')

        return accuracy, precision, recall, f1, pr_auc, auc_roc


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
            predicted = torch.argmax(outputs, dim=1) + 1
            # print(f" after predicted{predicted}")
            # print(f"after actual{targets}")
            correct_predictions += (predicted == targets).sum().item()
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
        accuracy, precision, recall, f1, pr_auc, auc_roc = compute_metrics(y_true, y_pred)

        # Ausgabe von Verlust und Metriken
        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, PR-AUC: {pr_auc:.4f}, "
              f"ROC-AUC: {auc_roc:.4f}")

        return epoch_loss, epoch_accuracy


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
                # Test für Vergleichbarkeit bei Berechnung der loss_fn  print(f"Label: {targets}")
                # Test für Vergleichbarkeit bei Berechnung der loss_fn  print(f"Vorhersage: {outputs}")
                loss = loss_fn(outputs, targets)

                # Berechne die Genauigkeit
                predicted = torch.argmax(outputs, dim=1) + 1
                # Test für Vergleichbarkeit bei Berechnung der loss_fn print(f"Vorhersage: {predicted}")
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

            # Berechne die Metriken
            accuracy, precision, recall, f1, pr_auc, auc_roc = compute_metrics(y_true, y_pred)

            # Gib den Verlust und die Metriken aus
            print(f"Validation: vLoss: {epoch_loss:.4f}, vAccuracy: {epoch_accuracy:.4f}, "
                  f"vPrecision: {precision:.4f}, vRecall: {recall:.4f}, "
                  f"vF1-Score: {f1:.4f}, vPR-AUC: {pr_auc:.4f},vROC-AUC: {auc_roc:.4f}")

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

    """
    # Splits aus der AnnotationsCSV nutzen:
    # Laden der Trainingsdaten
    train_data = fmamed.train_data
    print("Trainingsdaten erfolgreich geladen.")

    # Laden der Validierungsdaten
    val_data = fmamed.val_data
    print("Validierungsdaten erfolgreich geladen.")

    # Laden der Testdaten
    test_data = fmamed.test_data
    print("Testdaten erfolgreich geladen.")
    """

    # Erstelle Daten-Loader für Trainings-, Validierungs- und Testdaten
    print("Dataloader Trainingsdaten.")
    train_dataloader = create_data_loader(train_data, batch_size=BATCH_SIZE)
    print("Dataloader Validierungsdaten.")
    val_dataloader = create_data_loader(val_data, batch_size=BATCH_SIZE)
    print("Dataloader Testdaten.")
    test_dataloader = create_data_loader(test_data, batch_size=BATCH_SIZE)

    # Nutzen des vorgestalteten Pytorch VGG19
    print("vgg19 erstellen.")
    # VGG19 = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device)
    VGG19 = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device)
    # Einfrieren der Gewichte des vortrainierten Modells
    for param in VGG19.features.parameters():
        param.requires_grad = False
    # Ausgangsschicht auf 16 Features (Genre) anpassen:
    # Anzahl der Klassen definieren
    num_classes = 16
    # Ändern der siebten Schicht des Klassifikators
    VGG19.classifier[6] = nn.Linear(4096, num_classes)
    VGG19 = VGG19.to(device)
    print(f"{VGG19}")

    # Die Eingabeschicht des VGG19-Modells ändern, um mit den Spektrogramm-Eingabedaten umzugehen
    # print("Eingang des VGG19 auf Spektogramme in Tensor anpassen.")
    # VGG19.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).to(device)

    # initialisiere loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    # Weight Decay als L2-Regulierung als Maßnahme gegen Overfitting
    optimiser = torch.optim.Adam(VGG19.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # train model
    train(VGG19, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Testen Sie das Modell auf den Testdaten
    print("Testen des Modells...")
    test_loss, test_accuracy = validate(VGG19, test_dataloader, loss_fn, device)

    # Ausgabe der Ergebnisse
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

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
