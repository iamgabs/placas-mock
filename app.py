import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import string
import random

# Verifica se a GPU está disponível
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# Carregamento do dataset
with open("output_recortes.json", "r") as f:
    data = json.load(f)

letras_focadas = set(["W", "H", "I", "L", "O", "G", "Q", "D", "M", "N", "F", "E", "Y", "V", "R"])
letras_validas = list(string.ascii_letters)
char_to_idx = {char: idx for idx, char in enumerate(letras_validas)}
letras_data = {}

for key, values in data.items():
    numeros_filtrados = [
        item for item in values
        if item["char"] in letras_validas and "_min_7" not in item["imagem"]
    ]

    for item in numeros_filtrados:
        if item["char"].isdigit():
            print(f"Erro: número encontrado -> {item}")

    if numeros_filtrados:
        letras_data[key] = numeros_filtrados

with open("output_letras.json", "w") as f:
    json.dump(letras_data, f, indent=4)

# Transformações para treinamento
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
class DatasetLetras(Dataset):
    def __init__(self, data, transform=None):
        self.data = []
        self.transform = transform
        for key, values in data.items():
            for item in values:
                img_path = item["imagem"]
                label = item["char"]
                if label in letras_focadas:
                    self.data.append((img_path, char_to_idx[label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # Verifica se o arquivo existe antes de abrir
        if not os.path.exists(img_path):
            print(f"AVISO: Imagem não encontrada: {img_path}. Pulando...")
            # Escolher outra amostra aleatoriamente para substituir a ausente
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Erro ao abrir {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.data) - 1))

dataset = DatasetLetras(str_data, transform=transform_train)

"""## Divisão da base de dados para treinamento e validação"""

# dividindo o dataset em 80% para treinamento e 20% para validação
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

"""## Treinar a rede neural"""

# Função de treinamento para ResNet50
def treinar_resnet50(train_loader, val_loader, num_epochs=50, learning_rate=0.1):
    model = models.resnet50(weights=None)  # Usa ResNet50
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(letras_validas))
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_acc = 0.0
    print(f"Iniciando treinamento por {num_epochs} épocas...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        print(f"Época [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "melhor_modelo_resnet50_letras_geral.pth")
            print("Melhor modelo salvo!")
    print("Treinamento concluído!")
    return model

# Treinar modelo
modelo_treinado = treinar_resnet50(train_loader, val_loader, num_epochs=50, learning_rate=1e-3)
