import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epoch, model, train_loader, optimizer, loss_fn):
    model.train()
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y.long())
        loss.backward()
        optimizer.step()

        all_preds.append(pred.argmax(dim=1).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss.item()

def test(epoch, model, test_loader, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)
        loss = loss_fn(pred, batch.y.long())

        running_loss += loss.item() * batch.num_graphs
        all_preds.append(pred.argmax(dim=1).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return running_loss / len(test_dataset)

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

# Load datasets and create model
train_dataset = MoleculeDataset(root="data/", filename="HIV_train.csv")
test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv", test=True)
model = GNN(feature_size=train_dataset.num_features).to(device)

# Define optimizer, loss function, and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Create data loaders
NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

# Training loop
for epoch in range(1):
    train_loss = train(epoch, model, train_loader, optimizer, loss_fn)
    print(f"Epoch {epoch} | Train loss {train_loss}")

    if epoch % 5 == 0:
        test_loss = test(epoch, model, test_loader, loss_fn)
        print(f"Epoch {epoch} | Test loss {test_loss}")

    scheduler.step()

print("Training done..")