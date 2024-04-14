import torch
import random
import numpy as np
import torchaudio.transforms
from sklearn.preprocessing import label_binarize
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torchvision.models import VGG19_Weights, VGG19_BN_Weights
import torchvision.transforms as transforms
from FMA_Medium_MelSpecs import FreeMusicArchiveMedium
# from FMA_Medium_Data import FreeMusicArchiveMedium
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from sklearn.preprocessing import OneHotEncoder

# Konstanten
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
# L2-Regulierung / Norm-Penalisierung
WEIGHT_DECAY = 0.0001
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
IMAGE_DIR = "C:/AI_Datasets/fma_medium/hr_mel-spec-images/"
NUM_SAMPLES = 13219
SAMPLE_RATE = 22050
cep_lifter = 50
N_MFCC = 13
N_FTT = 2048
HOP_LENGTH = 512
N_MELS = 64
TRAIN_PERCENT = 0.5
VAL_PERCENT  = 0.25
TEST_PERCENT = 0.25


def split_data(dataset, train_percent=TRAIN_PERCENT, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT):
    # Berechne die Anzahl der Datenpunkte für jedes Split
    num_data = len(dataset)
    num_train = int(train_percent * num_data)
    num_val = int(val_percent * num_data)
    num_test = num_data - num_train - num_val
    print(f"Gesamt{num_data}, Train: {num_train}, Val{num_val}, Test{num_test} -> SUM {num_test+num_val+num_train}")
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
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Berechnen der ROC-AUC. Es ist wichtig anzumerken, dass roc_auc_score multiklassen-AUC für Sie berechnet.
    auc_roc = roc_auc_score(y_true_binarized, y_pred_binarized, average='weighted', multi_class='ovo')

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
        accuracy, precision, recall, f1, auc_roc = compute_metrics(y_true, y_pred)

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

            # Berechne die Metriken
            accuracy, precision, recall, f1, auc_roc = compute_metrics(y_true, y_pred)

            # Gib den Verlust und die Metriken aus
            print(f"\nValidation: vLoss: {epoch_loss:.4f}, vAccuracy: {epoch_accuracy:.4f}, "
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
    # BILDER
    # Definiere die Transformationen
    transformation = transforms.Compose([
        # transforms.PILToTensor(),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    """
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
    """

    fmamed = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                    IMAGE_DIR,
                                    transformation,
                                    device)
    print(f"{fmamed}")
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

    # Nutzen des vortraineirten Pytorch VGG19
    print("vgg19 erstellen.")
    VGG19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
    # print("Eingang des VGG19 auf Spektogramme in Tensor anpassen.")
    # VGG19.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Einfrieren der Gewichte des vortrainierten Modells
    # for param in VGG19.features.parameters():
    #     param.requires_grad = False

    # VGG19 Ausgangsschicht auf 16 Features (Genre) anpassen:
    # Anzahl der Klassen definieren
    num_classes = 16
    # Ändern der siebten Schicht des Klassifikators
    VGG19.classifier[6] = nn.Linear(4096, num_classes)
    model = VGG19.to(device)
    """
    # VGG19 Classifier für 16 Klassen anpassen:
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),  # Eingabegröße anpassen
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 12)  # Ausgabegröße anpassen
    )

    # Den angepassten Klassifikator der VGG19 hinzufügen
    VGG19.classifier = classifier
    model = VGG19.to(device)
    print(f"{VGG19}")
    """
    """
    # VGG19 Ausgangsschicht auf 16 Features (Genre) anpassen:
    # Anzahl der Klassen definieren
    num_classes = 16
    # Ändern der siebten Schicht des Klassifikators
    VGG19.classifier[6] = nn.Linear(4096, num_classes)
    VGG19 = VGG19.to(device)
    Neues Modell bauen:
    model = nn.Sequential()
    # Die Eingabeschicht des VGG19-Modells ändern, um mit den Spektrogramm-Eingabedaten umzugehen
    # Füge das vortrainierte VGG19-Modell hinzu
    model.add_module('base_model', VGG19)

    # Füge Flatten-Layer hinzu, um 3D-Tensor in 1D-Tensor umzuwandeln
    model.add_module('flatten', nn.Flatten())

    # Berechne die Eingabegröße für die Dense-Schicht
    num_features = VGG19.classifier[6].out_features

    # Füge Dense-Schicht mit 16 Ausgabeneuronen hinzu (entsprechend deinen Zielklassen)
    model.add_module('fc', nn.Linear(num_features,16))
    model.add_module('softmax', nn.Softmax(dim=1))  # Softmax-Aktivierungsfunktion für die Klassifikation
    
    print(f"{model}")
    model = model.to(device)
    """
    """
    # Die Klassen sind nicht balaciert, daher:
    class_weights = calculate_class_weights(train_dataloader.dataset)
    # initialisiere loss function + optimiser
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    """
    loss_fn = nn.CrossEntropyLoss()
    # Weight Decay als L2-Regulierung als Maßnahme gegen Overfitting
    optimiser = torch.optim.Adam(VGG19.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # train model
    train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Testen Sie das Modell auf den Testdaten
    print("Testen des Modells...\n")
    test_loss, test_accuracy = validate(VGG19, test_dataloader, loss_fn, device)

    # Ausgabe der Ergebnisse
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # save model
    torch.save(model.state_dict(), "VGG19_fma_med.pth")
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
