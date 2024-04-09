import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)
if __name__ == "__main__":
    # Annahme: output ist die Ausgabe des Modells und target sind die tatsächlichen Klassenlabels (nicht one-hot-kodiert)

    output = torch.tensor([[-1.3873e-01, -1.2026e-01, 1.6538e-01, 3.2585e-01, -4.4806e-01,
                            -2.3266e-01, 6.5503e-02, 7.6469e-02, -6.3794e-01, 5.8582e-01,
                            2.1140e-01, -5.5477e-01, 8.5655e-03, -1.4631e-02, 1.3079e-01,
                            2.2564e-01],
                           [-1.0680e-02, -6.4154e-01, -4.4990e-02, 5.9372e-01, -3.1779e-01,
                            1.4338e-02, 4.9265e-01, 2.7541e-02, -1.9794e-01, 1.0878e-01,
                            1.1618e-01, -3.5336e-01, 4.7742e-01, -3.5710e-02, 4.4545e-01,
                            1.1033e-01]])

    target = torch.tensor([1, 3])

    # Vorhersageklasse
    predicted_classes = torch.argmax(output, dim=1)

    # Überprüfung der Metriken
    accuracy = accuracy_score(target, predicted_classes)
    f1 = f1_score(target, predicted_classes, average='weighted')
    precision = precision_score(target, predicted_classes, average='weighted')
    recall = recall_score(target, predicted_classes, average='weighted')

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)