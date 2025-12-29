import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import SignLanguageDataset
from model import SignLSTM

# hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 50

# load data
dataset = SignLanguageDataset(root_dir='data')

if len(dataset) == 0:
    print("No data found. Please collect data before training.")
    exit()

# split dataset into train and validation
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# initialize model, loss function, optimizer
model = SignLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for sequence, labels in train_loader:
        # sequence shape: (batch, 30, 63)
        # labels shape: (batch)

        optimizer.zero_grad()
        outputs = model(sequence)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for sequence, labels in test_loader:
        outputs = model(sequence)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# save the model
torch.save(model.state_dict(), 'sign_language_model.pth')
print("Model saved as sign_language_model.pth")
