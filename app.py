# importações
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import zipfile

with open("output_recortes.json", "r") as f:
    data = json.load(f)

# separar o datase em outro dataset, de apenas números 0-9
numeros_data = {}

numeros_validos = {str(i) for i in range(10)}

for key, values in data.items():
    numeros_filtrados = [
        item for item in values
        if item["char"] in numeros_validos
        and "antigas" in item["imagem"]
        and "_min_7" not in item["imagem"]
    ]
    if numeros_filtrados:
        numeros_data[key] = numeros_filtrados

with open("output_numeros.json", "w") as f:
    json.dump(numeros_data, f, indent=4)

print("Novo dataset contendo apenas números salvo como 'output_numeros.json'")

with open("output_numeros.json", "r") as f:
    num_data = json.load(f)


# treinamento
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class DatasetNumerico(Dataset):
    def __init__(self, data, transform=None):
        self.data = []
        self.transform = transform
        for key, values in data.items():
            for item in values:
                img_path = item["imagem"]
                label = int(item["char"])
                self.data.append((img_path, label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


dataset = DatasetNumerico(num_data, transform=transform_train)

# dividindo o dataset em 80% para treinamento e 20% para validação
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def treinar_resnet18(train_loader, val_loader, num_epochs=30, learning_rate=1e-4, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar a ResNet18 pré-treinada
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # Ajustar a camada final com ReLU e Dropout
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 10)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_acc = 0.0
    print(f"Iniciando treinamento por {num_epochs} épocas...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        print(f"Época [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "melhor_modelo_resnet18.pth")
            print("Melhor modelo salvo!")
    print("Treinamento concluído!")
    return model

# Modelo treinado com 18 épocas
modelo_treinado = treinar_resnet18(train_loader, val_loader, num_epochs=18, learning_rate=0.005)

transform_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

