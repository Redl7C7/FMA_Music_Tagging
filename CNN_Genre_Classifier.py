from collections import Counter
import seaborn as sns
import torch
import torch.nn as nn
import random
import numpy as np
import torchaudio.transforms
import torchsampler
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from torch import nn
from tqdm import tqdm
from torch.utils.data import Subset, dataset, Dataset, SubsetRandomSampler
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import torchvision.models as models
from torchvision.models import VGG19_Weights, VGG19_BN_Weights, ResNeXt101_32X8D_Weights, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from FMA_Medium_ImageDataset import FreeMusicArchiveMedium
# from FMA_Medium_Data import FreeMusicArchiveMedium
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# Konstanten
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.00001
# L2-Regulierung / Norm-Penalisierung
WEIGHT_DECAY = 0.00001
FREEZE = True
ANNOTATIONS_FILE = 'C:/AI_Datasets/Tracks_Medium.csv'
MFCC_IMAGE_DIR = "C:/AI_Datasets/fma_medium/bunt-mfcc-images/"
MEL_SPEC_IMAGE_DIR = "C:/AI_Datasets/fma_medium/mel-spec-images/"
HR_MEL_SPEC_IMAGE_DIR = "C:/AI_Datasets/fma_medium/hr-mel-spec-images/"
LOG_MEL_SPEC_IMAGE_DIR = "C:/AI_Datasets/fma_medium/log_mel-spec-images/"
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 0.1
# vorbereitete Glob Vars für Auswertung:
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1 = []
val_f1 = []
train_roc_auc = []
val_roc_auc = []
train_recall = []
val_recall = []
train_precision = []
val_precision = []
class_Names = {'Classical',
               'Electronic',
               'Experimental',
               'Folk',
               'Hip-Hop',
               'Instrumental',
               'International',
               'Jazz',
               'Old-Time / Historic',
               'Pop',
               'Rock'}

def split_data(dataset, train_percent=TRAIN_PERCENT, val_percent=VAL_PERCENT, test_percent=TEST_PERCENT):
    # Berechne die Anzahl der Datenpunkte für jedes Split
    num_data = len(dataset)
    num_train = int(train_percent * num_data)
    num_val = int(val_percent * num_data)
    num_test = num_data - num_train - num_val
    # print(f"Gesamt{num_data}, Train: {num_train}, Val{num_val}, Test{num_test} -> SUM {num_test + num_val +
    # num_train}")
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

    # Berechnen der ROC-AUC. Es ist wichtig anzumerken, dass roc_auc_score Multiklassen-AUC für Sie berechnet.
    auc_roc = roc_auc_score(y_true_binarized, y_pred_binarized, average='macro', multi_class='ovr')

    return accuracy, precision, recall, f1, auc_roc


def compute_confusion_matrx(y_true, y_pred):
    class_names = {'Classical',
                   'Electronic',
                   'Experimental',
                   'Folk',
                   'Hip-Hop',
                   'Instrumental',
                   'International',
                   'Jazz',
                   'Old-Time / Historic',
                   'Pop',
                   'Rock'}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Plotte die Konfusionsmatrix als Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Wahre Klasse')
    plt.title('Konfusionsmatrix')
    plt.show()
    return conf_matrix


def create_data_loader(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, actual_epoch, last_epoch):
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

        # Berechnen der Metriken
        accuracy, precision, recall, f1, auc_roc = compute_metrics(y_true, y_pred)

        # Werte für Auswertung erhalten:
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_f1.append(f1)
        train_roc_auc.append(auc_roc)
        train_recall.append(recall)
        train_precision.append(precision)

        # Ausgabe der Klassenverteilung nach jeder Epoche
        print("Class distribution in the current epoch:", class_distribution)

        # Ausgabe von Verlust und Metriken
        print(f"\nLoss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {auc_roc:.4f}")

        # nach der letzten Epoche Konfusionsmatrix erzeugen:
        if actual_epoch == last_epoch:
            compute_confusion_matrx(y_true, y_pred)


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
        total_samples += 1

    # Berechne die Gewichte entsprechend der Klassenanzahl
    for label, count in class_counts.items():
        weight = total_samples / (count * len(class_counts))
        class_weights[label] = weight
        print(f"Klasse: {label} enthält {count} Samples und erhält Gewichtung: {weight}")
    weight_list = [class_weights[label] for label in sorted(class_weights.keys())]
    return weight_list


def validate(model, data_loader, loss_fn, device, actual_epoch, last_epoch):
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
            print(f"\nValidation: \nvLoss: {epoch_loss:.4f}, vAccuracy: {epoch_accuracy:.4f}, "
                  f"vPrecision: {precision:.4f}, vRecall: {recall:.4f}, "
                  f"vF1-Score: {f1:.4f}, vROC-AUC: {auc_roc:.4f}")

            # nach der letzten Epoche Konfusionsmatrix erzeugen:
            if actual_epoch == last_epoch:
                compute_confusion_matrx(y_true, y_pred)

            return epoch_loss, epoch_accuracy


def train(model, train_data_loader, val_data_loader, loss_fn, optimiser, device, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_accuracy = train_single_epoch(model, train_data_loader, loss_fn, optimiser, device, epoch, epochs)
        print("---------------------------")
        val_loss, val_accuracy = validate(model, val_data_loader, loss_fn, device, epoch, epochs)
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
    ])
    # Dataset mit den hochaufgelösten kontrastreichen Mel-Spektogrammen
    fmamed_hr_mel_specs = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                                 HR_MEL_SPEC_IMAGE_DIR,
                                                 transformation,
                                                 device)
    # Datasset mit den schlechter aufgelösten Mel-Spektogrammen und weniger Kontrasten
    """
    fmamed_mel_specs = FreeMusicArchiveMedium(ANNOTATIONS_FILE,
                                              MEL_SPEC_IMAGE_DIR,
                                              transformation,
                                              device)
    fmamed = ConcatDataset([fmamed_mel_specs, fmamed_hr_mel_specs])
    """
    fmamed = fmamed_hr_mel_specs
    # print(f"{fmamed}")
    """
    # Die Klassen sind nicht balanciert, daher werden Klassen je nach Repräsentation gewichtet:
    class_weights = calculate_class_weights(fmamed)  # vorher train_dataloader.dataset
    # initialisiere loss function + optimiser mit Klassengewichten
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    """
    loss_fn = nn.CrossEntropyLoss()

    # Verwende die Funktion split_data, um die Daten aufzuteilen
    print("Erstelle Trainings-, Test- und Validierungsdaten...")
    train_data, val_data, test_data = split_data(fmamed)


    # Berechnen der Gewichte für das Undersampling
    # train_class_weights = calculate_class_weights(train_data)
    def _subset_to_tensordataset(subset):
        subset_data = [sample[0] for sample in subset]
        subset_labels = [sample[1] for sample in subset]

        # Konvertiere die Daten und Labels in Tensoren
        data_tensor = torch.stack(subset_data)
        labels_tensor = torch.tensor(subset_labels)
        # Erstelle ein TensorDataset aus den Tensoren
        tensor_dataset = TensorDataset(data_tensor, labels_tensor)
        return tensor_dataset


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

    # train_dataloader = create_data_loader(train_data, batch_size=BATCH_SIZE)
    print("Dataloader Validierungsdaten.")
    val_dataloader = create_data_loader(_subset_to_tensordataset(val_data), batch_size=BATCH_SIZE)
    print("Dataloader Testdaten.")
    test_dataloader = create_data_loader(_subset_to_tensordataset(test_data), batch_size=BATCH_SIZE)
    """
    # Nutze vortrainiertes ResNet50
    print("RESNET50 erstellen.")
    RN50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Gewichte einfrieren
    for param in RN50.parameters():
        param.requires_grad = False
    # Ausgabe schicht
    num_ftrs = RN50.fc.in_features
    RN50.fc = nn.Linear(num_ftrs, 11)  # 11 Klassen für die Ausgabe

    model = RN50.to(device)
    """
    # Nutzen des vortrainierten Pytorch VGG19
    # print("vgg19 erstellen.")
    VGG19 = models.vgg19(weights=VGG19_Weights.DEFAULT)

    # Einfrieren der Gewichte des vortrainierten Modells
    # if FREEZE:True
    for param in VGG19.features.parameters():
        param.requires_grad = False

    # VGG19 Ausgangsschicht auf 11 Features (Genre) anpassen:
    VGG19.classifier[6] = nn.Linear(4096, 11)
    model = VGG19.to(device)

    """
    #eigener VGG19 Classifier für 11 Klassen:
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),  # Eingabegröße anpassen
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(4096, 12)  # Ausgabegröße anpassen - 11 Klassen
    )
    # Den angepassten Klassifikator der VGG19 hinzufügen
    VGG19.classifier = classifier
    model = VGG19.to(device)
    print(f"{model}")
    """
    # print(f"{model}")

    # Weight Decay als L2-Regulierung als Maßnahme gegen Overfitting
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # train model
    train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

    # Erstellen der Diagramme
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 9))
    plt.suptitle(
        f'Hyperparameter: '
        f'Lernrate={LEARNING_RATE}, '
        f'Weight Decay={WEIGHT_DECAY}, '
        f'Batch Size={BATCH_SIZE}, '
        f'Epochen={EPOCHS},'
        f'on Log-Mel-Spectograms')
    # F1 Diagramm
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_f1, label='Train F1')
    plt.plot(epochs, val_f1, label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.title('F1 Score')
    plt.legend()

    # ROC-AUC Diagramm
    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_roc_auc, label='Train ROC-AUC')
    plt.plot(epochs, val_roc_auc, label='Val ROC-AUC')
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC Score')
    plt.legend()

    # Recall Diagramm
    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_recall, label='Train Recall')
    plt.plot(epochs, val_recall, label='Val Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall')
    plt.legend()

    # Precision Diagramm
    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_precision, label='Train Precision')
    plt.plot(epochs, val_precision, label='Val Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.legend()

    # Accuracy Diagramm
    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Verlustdiagramm
    plt.subplot(3, 2, 6)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()  # Für bessere Layout-Anpassung
    plt.show()

    # Testen Sie das Modell auf den Testdaten
    print("Testen des Modells...\n")
    test_loss, test_accuracy = validate(model, test_dataloader, loss_fn, device, 1, 1)

    # Ausgabe der Ergebnisse
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # save model
    torch.save(model.state_dict(), "VGG19_cm_fma_med.pth")
    print("Trainiertes Netz als cnn_fma_med.pth gespeichert.")
