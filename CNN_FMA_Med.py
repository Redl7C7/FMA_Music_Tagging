from torch import nn
from torchsummary import summary


class CNNetwork(nn.Module):

    # Architektur Konstruktor
    def __init__(self):
        super().__init__()
        # 4 conv Blocks / flatten results / linear layer / softmax Ausgabe für 16 Genre
        # Conv Blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Dense Layer (Linear Layer)
        # Num Out Chan vorhergehender Layer: 128 | Shape der Eingabe 5 * 4
        # -> Hängt von der Faltung ab, was am Ende reingeht
        # Out Features= 16 -> 16 Genre
        # Siehe torchsummary: MaxPool2d-12 -> Shape after conv4: torch.Size([9, 128, 5, 163])            0
        self.linear = nn.Linear(128*5*163, 16)
        # Softmax normaliseren
        self.softmax = nn.Softmax(dim=1)

    # Forwarding der Daten zwischen den Schichten
    def forward(self, input_data):
        # print("Shape before conv1:", input_data.shape)
        x = self.conv1(input_data)
        # print("Shape after conv1:", x.shape)
        x = self.conv2(x)
        # print("Shape after conv2:", x.shape)
        x = self.conv3(x)
        # print("Shape after conv3:", x.shape)
        x = self.conv4(x)
        # print("Shape after conv4:", x.shape)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
if __name__=="__main__":
    cnn = CNNetwork()
    summary(cnn.cuda(), (1, 64, 44))
