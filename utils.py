#!/usr/bin/env python

# Importing necessary libraries
import os
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import math
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# Function to adjust learning rate during training
def adjust_learning_rate(optimizer, curr_epoch, warmup_epochs, lr, min_lr, num_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if curr_epoch < warmup_epochs:
        lr = lr * curr_epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (curr_epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# Function to read eye images from the JustRAIGS dataset
def read_eye_image_justraigs(root, sub_dir, eye_id):
    """
    Lit une image en fonction du sous-répertoire (sub_dir) et de l'identifiant de l'œil (eye_id).

    Args:
        sub_dir (str): Le sous-répertoire contenant l'image (ex : "JustRaigs_train_0/0").
        eye_id (str): L'identifiant de l'œil (ex : "Train000000").

    Returns:
        numpy.ndarray: L'image lue au format NumPy.
    """
    # Définir le chemin complet du répertoire
    directory_path = os.path.join(root, sub_dir) 

    # Vérifier l'existence du répertoire
    if not os.path.exists(directory_path):
        print(f"Le répertoire n'existe pas : {directory_path}")
        return None

    # Liste des extensions d'image à essayer
    image_extensions = ['.jpg', '.png', '.JPG', '.PNG', '.JPEG',]

    for ext in image_extensions:
        # Définir le chemin complet du fichier image avec l'extension actuelle
        file_path = os.path.join(directory_path, f"{eye_id}{ext}")

        if os.path.exists(file_path):
            try:
                # Lire l'image en utilisant OpenCV
                image = cv2.imread(file_path)

                if image is None:
                    raise FileNotFoundError(f"Impossible de lire l'image : {file_path}")

                return image

            except Exception as e:
                print(f"Une erreur s'est produite : {str(e)}")
                return None

    # Si aucune extension n'a fonctionné
    print(f"Aucune image trouvée pour {eye_id} dans le répertoire {sub_dir}")
    return None

# Function to compute reference performance (P_ref) based on desired specificity
def compute_P_ref(fpr, tpr, thresholds):
    # Desired specificity
    desired_specificity = 0.95

    # Find the index of the threshold that is closest to the desired specificity
    idx = np.argmax(fpr >= (1 - desired_specificity))

    # Get the corresponding threshold
    threshold_at_desired_specificity = thresholds[idx]

    print(f"Threshold at Specificity {desired_specificity*100:.2f}%: {threshold_at_desired_specificity:.4f}")

    # Get the corresponding TPR (sensitivity)
    sensitivity_at_desired_specificity = tpr[idx]

    print(f"Sensitivity at Specificity {desired_specificity*100:.2f}%: {sensitivity_at_desired_specificity:.4f}")
    
    return sensitivity_at_desired_specificity


# ### DATASET

# Dataset class for task 1 of the JustRAIGS dataset
class RAIGS_task1_ds(Dataset):
    def __init__(self,
                data_info,
                db_root,
                mode='train',
                resize=True):
        
        self.data_info = data_info
        self.db_root = db_root
        self.mode = mode
        self.resize = resize
        self.data_info.reset_index(drop=True, inplace=True)
        label_list = self.data_info["Label_INT"].tolist()
        self.label_list = torch.nn.functional.one_hot(torch.as_tensor(label_list)).float()
    
    def __getitem__(self, idx):

        patient_id = self.data_info["Patient ID"][idx]
        sub_dir = self.data_info["sub_dir"][idx]
        eye = self.data_info["Eye ID"][idx]
        label = self.label_list[idx]
                
        img_path = os.path.join(self.db_root, sub_dir, eye + ".jpg" )  
        img = Image.open(img_path)
        
        if self.resize : 
            new_size = (800,800)
            img = img.resize(new_size, resample=0)
            
        if self.mode == "train":
            im_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            ])
            img = im_aug(img)
        
        img = transforms.PILToTensor()(img)

        if self.mode == 'test':
            return img, label, patient_id + eye

        if self.mode == "train" or self.mode == "validation" :           
            return img, label

    def __len__(self):
        return len(self.data_info)


# Dataset class for task 2 of the JustRAIGS dataset
class RAIGS_task2_ds(Dataset):
    def __init__(self, data_info, db_root, mode='train', resize=True, image_size=384):
        self.data_info = data_info[data_info.set == mode]
        self.db_root = db_root
        self.mode = mode
        self.resize = resize
        self.image_size = image_size
        self.data_info.reset_index(drop=True, inplace=True)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        img_path = os.path.join(self.db_root, row["sub_dir"], row["Eye ID"] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        
        labels = np.array([row['ANRS'], row['ANRI'], row['RNFLDS'], row['RNFLDI'],
                           row['BCLVS'], row['BCLVI'], row['NVT'], row['DH'], row['LD'], row['LC']], dtype=np.float32)
        
        if self.resize:
            img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        
        img = np.array(img)
        
        img = self.transform(img)

        return img, labels

    def transform(self, img):
        if self.mode == "train":
            transform = A.Compose([
                A.Flip(),
                A.ShiftScaleRotate(),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ], p=0.5),
                A.CoarseDropout(max_height=int(self.image_size * 0.05), max_width=int(self.image_size * 0.8), p=0.5),
                A.OneOf([
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
                ], p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        return transform(image=img)['image']

    def __len__(self):
        return len(self.data_info)
