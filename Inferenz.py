import torch
# Aufrufe aus main
from main import FeedForwardNet, download_mnist_datasets

class_mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def predict(model, input, target, class_mapping):
    # Modell evaluieren
    model.eval()
    # No_Grad
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1 -> Beispielanzahl, 10 -> Klassen) -> [ [0.1, 0.01, ..., 0.6] ]
        # Index vorhersagen mit dem höchsten Wert, letzter Kommentar Index bei 0.6
        predicted_index = predictions[0].argmax(0)
        # vorhergesagten Index mappen
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # Modell aus Training in "main" laden
    feed_forward_net = FeedForwardNet()
    # Parameter laden
    state_dict = torch.load("feedforwardnet.pth")
    # Parameter in Modell setzen
    feed_forward_net.load_state_dict(state_dict)

    # MNIST validation dataset laden
    _, validation_data = download_mnist_datasets()

    # Ein Beispiel aus dem Validierungsdatensatz für Inferenz laden
    input, target = validation_data[0][0], validation_data[0][1]

    # Inferenz zwischen erwartetem Ergebnis und dem vorhergesagtem Ergebnis
    predicted, expected = predict(feed_forward_net, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")