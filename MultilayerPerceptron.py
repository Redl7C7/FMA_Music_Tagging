import numpy as np

# Speichern der Aktivierungen und der Derivate in MLP-Klasse

# Implementierung Backpropagation

# absteigende Gradienten implementieren

# Training implementieren

# Training mit Dummy Dataset

# Vorhersagen treffen

#Multilayer Perceptron anlegen
class MLP:
    # Default definieren
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        #Initialisierung: zufällige Gewichtung zum Start
        self.weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])
            self.weights.append(w)

        # Aktivierung und Derivate
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers(i))
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers)-1)
            d = np.zeros((layers[i], layers[i+1]))
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

    def back_propagate(self, error):
        # Fehler des Output-Layers an den Input Layer geben
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            current_activations = self.activations[i] # -->

            self.derivatives[i] = np.dot(current_activations, delta)


    def _sigmoid_derivative(selfself, x):
        return x * (1.0 - x)


    #Sigmoid Fkt implementieren
    def _sigmoid(self, x):
        return 1 / (1+np.exp(-x))
if __name__ == "__main__":
    # MLP erstellen
    mlp = MLP()

    # Inputs anlegen Vektor
    inputs = np.random.rand(mlp.num_inputs)

    # Forward Propagation ausführen
    outputs = mlp.forward_propagate(inputs)

    # Ergebnisse anzeigen
    print("Input is: {}".format(inputs))
    print("Output is: {}".format(outputs))