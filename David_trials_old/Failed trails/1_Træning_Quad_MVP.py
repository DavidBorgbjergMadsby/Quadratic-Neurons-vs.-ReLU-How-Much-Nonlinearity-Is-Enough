import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# --- 1. DEFINER DIN KVADRATISKE NEURON ---
class QuadraticLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuadraticLinear, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight2 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(self.weight1)
        nn.init.zeros_(self.weight2)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return (
            torch.matmul(x, self.weight1.t())
            + torch.matmul(x**2, self.weight2.t())
            + self.bias
        )


if __name__ == "__main__":
    # --- 2. SETUP DEVICE OG DATA ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benytter enhed: {device}")

    transform = transforms.Compose(
        [
            transforms.Resize(224),  # ResNet er født til 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Download data (Husk download=True her, så den henter dem til G-bar)
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    # --- 3. MODEL OPSÆTNING ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = QuadraticLinear(num_ftrs, 10)
    model = model.to(device)  # FLYT MODEL TIL GPU

    # --- 4. OPTIMERING OG TRÆNING ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Træn kun det nye lag

    print("Starter træning...")
    model.train()
    for epoch in range(2):  # Start med 2 epoker for at se om det virker
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)  # FLYT DATA TIL GPU

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {loss.item():.3f}")

    print("Træning færdig!")
    torch.save(model.state_dict(), "resnet_quadratic.pth")
