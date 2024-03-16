import numpy as np
from random import random

# Speichern der Aktivierungen und der Derivate in MLP-Klasse

# Implementierung Backpropagation

# absteigende Gradienten implementieren

# Training implementieren

# Training mit Dummy Dataset

# Vorhersagen treffen

#Multilayer Perceptron anlegen
class MLP(object):
    # Default definieren
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        #Initialisierung: zufällige Gewichtung zum Start
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights

        # Aktivierung je Layer speichern
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # Ableitung je Layer speichern
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        activations = inputs
        self.activations[0] = inputs
        for i, w in enumerate(self.weights):
            # Inputs berechnen np.dot Matrixoperation
            net_inputs = np.dot(activations, w)
            # Aktivierung berechnen mit Sigmoid Fkt
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations

        return activations

    def back_propagate(self, error, verbose=False):
        # Fehler des Output-Layers an den Input Layer geben
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations) # --> Aus Array einen vertikalen Vektor machen
            delta_reshaped =  delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # --> Aus Array einen vertikalen Vektor machen
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            if verbose:
                print("Ableitungen für W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate): # Iteration durch alle Gewichte
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        # Epochen --> Häufigkeit, wie oft der Trainingsdatensatz durch das Netzwerk verarbeitet werden
        for i in range(epochs):
            sum_errors = 0
            for input, target in zip(inputs, targets):
                # Forward Propagation ausführen
                output = mlp.forward_propagate(input)

                # Fehler berechnen
                error = target - output

                # Back Propagation ausführen
                self.back_propagate(error)

                # Absteigende Gradienten anwenden
                self.gradient_descent(learning_rate)

                sum_errors += self._mse(target, output) # _mse = Mean Squared Error
            # Bericht über Fehler je Epoche --> Verbesserung über sinkende Fehler
            print("Error: {} bei Epoche {}".format(sum_errors / len(inputs), i))
        # Abschluss benachrichtigen
        print("Training complete!")
        print("=====")
    def _mse(self, target, output):
        return np.average((target - output)**2)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)


    #Sigmoid Fkt implementieren
    def _sigmoid(self, x):
        return 1 / (1+np.exp(-x))

# Debug Script
if __name__ == "__main__":
    # create a dataset to train a network for the sum operation
    items = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))