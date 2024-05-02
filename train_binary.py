import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import torch
import timm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc
from utils import *
from sklearn import metrics

# Set up argument parsing
parser = argparse.ArgumentParser(description="Train a deep learning model for image classification")
parser.add_argument("--model_name", type=str, default="resnet50", help="Model name as specified by TIMM library")
parser.add_argument("--val_fold", type=int, default=0, help="Validation fold number")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--init_epochs", type=int, default=129, help="Initial number of epochs for training")
parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
parser.add_argument("--num_epochs", type=int, default=250, help="Total number of epochs for training")
parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
parser.add_argument("--base_lr", type=float, default=1e-3, help="Base learning rate")
parser.add_argument("--data_root", type=str, default="/path/to/dataset", help="Root directory of the dataset")
parser.add_argument("--csv_path", type=str, default="JustRAIGS_data_resampling_fold_all.csv", help="Path to the CSV file containing labels")
args = parser.parse_args()

# Check if CUDA is available and set device accordingly
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assign parsed arguments
model_name = args.model_name
val_fold = args.val_fold
batch_size = args.batch_size
init_epochs = args.init_epochs
warmup_epochs = args.warmup_epochs
num_epochs = args.num_epochs
min_lr = args.min_lr
b_lr = args.base_lr
db_root = args.data_root
label_path = args.csv_path

# Calculate learning rate based on batch size
lr = b_lr * batch_size / 256
print("base lr: %.2e" % (b_lr * 256 / batch_size))
print("actual lr: %.2e" % lr)

torch.manual_seed(0)

# Load and preprocess labels
data = pd.read_csv(label_path)
label_mapping = {"NRG": 0, "RG": 1}
data['Label_INT'] = data['Label'].map(label_mapping)

# Split data by labels and fold
data_RG = data[data.Label == "RG"]
data_NRG = data[data.Label == "NRG"]
data_train_rg = data_RG[data_RG.Fold != val_fold]
data_val_rg = data_RG[data_RG.Fold == val_fold]
data_train_nrg = data_NRG[data_NRG.Fold != val_fold]
data_val_nrg = data_NRG[data_NRG.Fold == val_fold]

# Create balanced datasets for training and validation
val_df = pd.concat([data_val_rg, data_val_nrg[:len(data_val_rg) * 3]], ignore_index=True)
train_df = pd.concat([data_train_rg, data_train_nrg[:len(data_train_rg) * 3]], ignore_index=True)

print('train nombre total :',len(train_df))
print('train nombre RG :',len(data_train_rg))
print('val nombre total :',len(val_df))
print('val nombre RG :',len(data_val_rg))

# Initialize datasets
train_dataset = RAIGS_task1_ds(data_info=train_df, db_root=db_root, mode='train')
val_dataset = RAIGS_task1_ds(data_info=val_df, db_root=db_root, mode='validation')

# Initialize data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=6, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True, num_workers=6,
                        pin_memory=True)

# Training preparation
summary_dir = './logs'
torch.backends.cudnn.benchmark = True
summaryWriter = SummaryWriter(summary_dir)
print('cuda',torch.cuda.is_available())
print('gpu number',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

# Initialize model, loss function, and optimizer
model = timm.create_model(model_name, pretrained=True, num_classes=2, in_chans=3).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)

# Training loop
val_interval = 1
best_metric1 = -1 #acc
best_metric1_epoch = -1
metric1_values = []
epoch_loss_values = []
best_metric2 = -1 #auc
best_metric2_epoch = -1
metric2_values = []
best_metric3 = -1 #pref
best_metric3_epoch = -1
metric3_values = []

filepath = './weights'
if not os.path.exists(filepath):
    os.makedirs(filepath)

best_model_auc_path = './weights/bestmodelauc.pth'
best_model_pref_path = './weights/bestmodelpref.pth'

if init_epochs > 0:
    model.load_state_dict(torch.load(best_model_pref_path))
    model.cuda()

for epoch in range(init_epochs, num_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    train_lr = 0.
    
    for num_batch, batch_data in enumerate(train_loader):
        step += 1
        inputs, labels = batch_data[0].float().to(device), batch_data[1].to(device)
        curr_lr = adjust_learning_rate(optimizer, num_batch / len(train_loader) + (epoch +1) , warmup_epochs, lr , min_lr, num_epochs)
        train_lr += curr_lr
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}, LR: {train_lr/len(train_loader)}")
    summaryWriter.add_scalar("train_loss", epoch_loss, epoch + 1)

    # Validation process
    if (epoch + 1) % val_interval == 0:
        model.eval()
        y_targets = []
        y_predictions = []
        num_correct = 0.0
        metric1_count = 0
                
        for val_data in val_loader:
            val_images, val_labels = val_data[0].float().to(device), val_data[1].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                metric1_count += len(value)
                num_correct += value.sum().item()
                y_targets.extend(val_labels.cpu().detach().numpy().tolist())
                y_predictions.extend(torch.softmax(val_outputs,dim=1).cpu().detach().numpy().tolist())

        # Calculate accuracy and AUC
        metric1 = num_correct / metric1_count
        metric2 = metrics.roc_auc_score(y_targets, y_predictions)
        metric2_values.append(metric2)
        
        if metric2 > best_metric2:
            best_metric2 = metric2
            best_metric2_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_auc_path)
        
        # Calculate sensitivity at a fixed specificity
        y_targets_ = [element[1] for element in y_targets]
        y_predictions_ = [element[1] for element in y_predictions]
        fpr, tpr, thresholds = roc_curve(y_targets_, y_predictions_)
        metric3 = compute_P_ref(fpr,tpr,thresholds)
        metric3_values.append(metric3)
        
        if metric3 > best_metric3:
            best_metric3 = metric3
            best_metric3_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_pref_path)
            if metric3 > 0.8:
                path = './weights/model_pref_' + str(epoch) + '_'+ str(metric3) + '.pth'
                torch.save(model.state_dict(), path)
        
        print(f"Current epoch: {epoch+1} current auc: {metric2:.4f} current acc: {metric1:.4f} current pref: {metric3:.4f}")
        print(f"Best auc: {best_metric2:.4f} at epoch {best_metric2_epoch}")
        print(f"Best pref: {best_metric3:.4f} at epoch {best_metric3_epoch}")
        
        summaryWriter.add_scalar("val_auc", best_metric2, epoch +1)
                  
print(f"Training completed, best_metric (auc): {best_metric2:.4f} at epoch: {best_metric2_epoch}")
summaryWriter.close()

# Testing and results visualization
print('############# TRAIN FINISH，START TEST ################')
best_model3_path = './weights/bestmodelpref.pth'
model = timm.create_model(model_name, pretrained=False, num_classes=2, in_chans=3)
model.load_state_dict(torch.load(best_model3_path))
model.cuda()

model.eval()
y_targets = []
y_predictions = []

for i,test_data in enumerate(val_loader):
    test_images, test_labels = test_data[0].float().to(device), test_data[1].to(device)
    with torch.no_grad():
        test_outputs = model(test_images)
        y_targets.extend(test_labels.cpu().detach().numpy().tolist())
        y_predictions.extend(torch.softmax(test_outputs,dim=1).cpu().detach().numpy().tolist())

metric2 = metrics.roc_auc_score(y_targets, y_predictions)
print("AUC on data test = ", metric2)

# Plot and save ROC curve
dirpath = './plot'
if not os.path.exists(dirpath):
    os.makedirs(dirpath)
    
y_targets_ = [element[1] for element in y_targets]
y_predictions_ = [element[1] for element in y_predictions]
fpr, tpr, thresholds = roc_curve(y_targets_, y_predictions_)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(dirpath,'roc_curve.png'), format='png')
plt.show()

# Find sensitivity at a specified specificity
desired_specificity = 0.95
idx = np.argmax(fpr >= (1 - desired_specificity))
threshold_at_desired_specificity = thresholds[idx]
sensitivity_at_desired_specificity = tpr[idx]

print(f"Threshold at Specificity {desired_specificity*100:.2f}%: {threshold_at_desired_specificity:.4f}")
print(f"Sensitivity at Specificity {desired_specificity*100:.2f}%: {sensitivity_at_desired_specificity:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[idx], tpr[idx], c='red', marker='o', label=f'Sensitivity (at Spec. {desired_specificity*100:.2f}%) = {sensitivity_at_desired_specificity:.4f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(dirpath,'roc_curve2.png'), format='png')
plt.show()

print('############# TEST FINISH _________ ################')
