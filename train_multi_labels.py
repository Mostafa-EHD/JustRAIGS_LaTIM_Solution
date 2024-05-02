import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import hamming_loss
import timm
from utils import adjust_learning_rate, RAIGS_task2_ds  # Assuming you have a separate utility file

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train a multilabel classification model with multiple backbones")
parser.add_argument("--backbone", type=str, default="resnet50", choices=["deit3_base_patch16_384", "eva_large_patch14_336", "resnet50"], help="Backbone for the model")
parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs for adjusting learning rate")
parser.add_argument("--data_root", type=str, default="/path/to/data", help="Root path to the dataset")
parser.add_argument("--csv_path", type=str, default="JustRAIGS_data_resampling_final_only_RG.csv", help="Path to the label CSV file")
args = parser.parse_args()

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess DataFrame
df = pd.read_csv(args.csv_path)

# Define model based on the selected backbone
class Task2_model(nn.Module):
    def __init__(self, backbone):
        super(Task2_model, self).__init__()
        if backbone == "resnet50":
            self.model = timm.create_model(backbone, pretrained=False, num_classes=0, in_chans=3)
            self.fc = nn.Linear(2048, 10)  # Adjust according to the output features of the backbone
        elif backbone == "deit3_base_patch16_384":
            self.model = timm.create_model(backbone, pretrained=True, num_classes=0, in_chans=3)
            self.fc = nn.Linear(768, 10)  # Adjust according to DeiT's output features
        elif backbone == "eva_large_patch14_336":
            self.model = timm.create_model(backbone, pretrained=True, num_classes=0, in_chans=3)
            self.fc = nn.Linear(1024, 10)  # Adjust according to EVA's output features

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# Instantiate model, optimizer, and loss functions
model = Task2_model(backbone=args.backbone).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
bce_loss_function = nn.BCEWithLogitsLoss()

# Training loop
def train(model, optimizer, num_epochs, train_loader, val_loader, warmup_epochs):
    summary_dir = './logs'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summaryWriter = SummaryWriter(summary_dir)
    best_hamming = float('inf')
    best_bce = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_hamming = 0

        for batch_data in train_loader:
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = bce_loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = (outputs.sigmoid() > 0.5).float()
            hamming = hamming_loss(predictions.cpu().detach().numpy(), labels.cpu().detach().numpy())
            total_hamming += hamming

        avg_loss = total_loss / len(train_loader)
        avg_hamming = total_hamming / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Hamming Loss: {avg_hamming:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_hamming = 0
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += bce_loss_function(val_outputs, val_labels).item()
                val_predictions = (val_outputs.sigmoid() > 0.5).float()
                val_hamming += hamming_loss(val_predictions.cpu().numpy(), val_labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_hamming /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}, Validation Hamming Loss: {val_hamming:.4f}')

            if val_hamming < best_hamming:
                best_hamming = val_hamming
                torch.save(model.state_dict(), 'best_model_hamming.pth')
                print('Updated best hamming loss model.')

            if val_loss < best_bce:
                best_bce = val_loss
                torch.save(model.state_dict(), 'best_model_bce.pth')
                print('Updated best BCE loss model.')

        summaryWriter.add_scalar('Loss/train', avg_loss, epoch)
        summaryWriter.add_scalar('Hamming Loss/train', avg_hamming, epoch)
        summaryWriter.add_scalar('Loss/val', val_loss, epoch)
        summaryWriter.add_scalar('Hamming Loss/val', val_hamming, epoch)
        
        # Adjust learning rate
        curr_lr = adjust_learning_rate(optimizer, epoch, warmup_epochs, args.lr, args.min_lr, num_epochs)


    summaryWriter.close()

# Main execution
if __name__ == "__main__":
    
    image_size_map = {
        "resnet50": 800,
        "deit3_base_patch16_384": 384,
        "eva_large_patch14_336": 336
    }

    train_dataset = RAIGS_task2_ds(data_info=df[df['set'] == "train"], 
                                db_root=args.data_root,
                                mode='train',
                                resize=True,
                                image_size=image_size_map[args.backbone])

    val_dataset = RAIGS_task2_ds(data_info=df[df['set'] == "validation"], 
                                db_root=args.data_root,
                                mode='validation',
                                resize=True,
                                image_size=image_size_map[args.backbone])

    test_dataset = RAIGS_task2_ds(data_info=df[df['set'] == "test"], 
                                db_root=args.data_root,
                                mode='test',
                                resize=True,
                                image_size=image_size_map[args.backbone])


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Start training
    train(model, optimizer, args.num_epochs, train_loader, val_loader, args.warmup_epochs)
