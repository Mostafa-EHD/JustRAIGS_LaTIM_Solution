import torch.nn as nn
import torch 
import os 
import timm

####################################################################################
################ Model Binary ######################################################
####################################################################################


class Model_ensemble(nn.Module):
    def __init__(self,load_weights,path1,path2,path3,path4,path5):
        super(Model_ensemble, self).__init__()

        self.modelf0 = timm.create_model("resnet50", pretrained=False, num_classes=2, in_chans=3)
        
        if load_weights == True:
            self.modelf0.load_state_dict(torch.load(path1,map_location=torch.device('cpu')))
        
        self.modelf1 = timm.create_model("resnet50", pretrained=False, num_classes=2, in_chans=3)
        
        if load_weights == True:
            self.modelf1.load_state_dict(torch.load(path2,map_location=torch.device('cpu')))
            
        self.modelf2 = timm.create_model("resnet50", pretrained=False, num_classes=2, in_chans=3)
        
        if load_weights == True:
            self.modelf2.load_state_dict(torch.load(path3,map_location=torch.device('cpu')))
            
        self.modelf3 = timm.create_model("resnet50", pretrained=False, num_classes=2, in_chans=3)
        
        if load_weights == True:
            self.modelf3.load_state_dict(torch.load(path4,map_location=torch.device('cpu')))
            
        self.modelf4 = timm.create_model("resnet50", pretrained=False, num_classes=2, in_chans=3)
        
        if load_weights == True:
            self.modelf4.load_state_dict(torch.load(path5,map_location=torch.device('cpu')))

    def forward(self, x):
        out1 = self.modelf0(x)[0]
        out2 = self.modelf1(x)[0]
        out3 = self.modelf2(x)[0]
        out4 = self.modelf3(x)[0]
        out5 = self.modelf4(x)[0]
        out = (out1+out2+out3+out4+out5)/5
#         print('out1=',out1)
#         print('out2=',out2)
#         print('out=',out)
        return out


class model_task1:
    def __init__(self):
        self.checkpoint = "./model/task1/bestmodelauc0.pth"
        self.checkpoint2 = "./model/task1/bestmodelauc1.pth"
        self.checkpoint3 = "./model/task1/bestmodelauc2.pth"
        self.checkpoint4 = "./model/task1/bestmodelauc3.pth"
        self.checkpoint5 = "./model/task1/bestmodelauc4.pth"
        
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cuda")
       

    def load(self):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        #self.model = Model_ensemble(load_weights=False)
        # join paths
        checkpoint_path = os.path.join(self.checkpoint)
        checkpoint_path2 = os.path.join(self.checkpoint2)
        checkpoint_path3 = os.path.join(self.checkpoint3)
        checkpoint_path4 = os.path.join(self.checkpoint4)
        checkpoint_path5 = os.path.join(self.checkpoint5)

        self.model = Model_ensemble(True,checkpoint_path,checkpoint_path2,checkpoint_path3, checkpoint_path4, checkpoint_path5)
        #self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images):
        
        with torch.no_grad():
            score = self.model(images)

        return score


####################################################################################
################ Model Multilabel ##################################################
####################################################################################

class model1(nn.Module):
    
    def __init__(self,backbone='eva_large_patch14_336', num_features = 1024): 
        super().__init__()
        
        self.encoder = timm.create_model(backbone,pretrained=False, num_classes=10, in_chans=3, embed_dim= 1024)
       
        
    def forward(self, image):
        x = self.encoder(image)
        
        return x
    
class model2(nn.Module):
    
    def __init__(self,backbone='deit3_base_patch16_384'): 
        super().__init__()
        
        self.encoder = timm.create_model(backbone,pretrained=False, num_classes=10, in_chans=3)
       
        
    def forward(self, image):
        x = self.encoder(image)
        
        return x    

class model3(nn.Module):
    
    def __init__(self,backbone='resnet50', num_features = 2048): 
        super().__init__()
        
        self.encoder = timm.create_model(backbone,pretrained=False, num_classes=0, in_chans=3)
        self.fc = nn.Linear(2048,10)
       
        
    def forward(self, image):
        x = self.encoder(image)
        x = self.fc(x)

        return x
    
class Model_ensemble2(nn.Module):
    def __init__(self,load_weights,path1,path2,path3):
        super(Model_ensemble2, self).__init__()

        self.model1 = model1()
        if load_weights == True:
            self.model1.load_state_dict(torch.load(path1,map_location=torch.device('cpu')))
        
        self.model2 = model2()
        if load_weights == True:
            self.model2.load_state_dict(torch.load(path2, map_location=torch.device('cpu')))
          
        self.model3 = model3()
        if load_weights == True:
            self.model3.load_state_dict(torch.load(path3, map_location=torch.device('cpu')))
              

    def forward(self, x1, x2, x3):
        out1 = self.model1(x1)
        out2 = self.model2(x2) 
        out3 = self.model3(x3) 
        #torch.softmax(val_pred_logit1,dim =1)

        #out = torch.min(torch.min(out1, out2),out3)
        out = (out1 + out2 + out3) / 3
        return out

class model_task2:
    def __init__(self):
        self.checkpoint = "./model/task2/bestmodel_eva.pth"
        self.checkpoint2 = "./model/task2/bestmodel_deit.pth"
        self.checkpoint3 = "./model/task2/bestmodel_resnet.pth"
        
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cuda")
       

    def load(self):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        #self.model = Model_ensemble(load_weights=False)
        # join paths
        checkpoint_path = os.path.join(self.checkpoint)
        checkpoint_path2 = os.path.join(self.checkpoint2)
        checkpoint_path3 = os.path.join(self.checkpoint3)

        self.model = Model_ensemble2(True,checkpoint_path,checkpoint_path2,checkpoint_path3)
        #self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x1, x2, x3):
        
        with torch.no_grad():
            output = self.model(x1, x2, x3)

        return output

import random

import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks

import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np 


def remove_black_borders(image):
    image_np = np.array(image)

    # Find all rows and columns where the image is not black
    non_black_rows = np.where(np.any(image_np > 10, axis=(1, 2)))[0]
    non_black_cols = np.where(np.any(image_np > 10, axis=(0, 2)))[0]

    # Get the bounds of the non-black areas
    row_min, row_max = non_black_rows[[0, -1]]
    col_min, col_max = non_black_cols[[0, -1]]

    # Crop the image using PIL's crop method
    cropped_image = image.crop((col_min, row_min, col_max, row_max))
    return cropped_image

def preprocess1(image ): #PIL IMAGE
    img = remove_black_borders(image)
    img = img.resize((800,800), resample=0)
    img = transforms.PILToTensor()(img)
    img = img.unsqueeze(0) #add batch
    return img #tensor

def preprocess2_1(image ): #PIL IMAGE
    img = remove_black_borders(image)
    img = img.resize((336,336), resample=0)
    
    img = transforms.PILToTensor()(img)
    
    img = img.float() / 255.0  # Converts to float and scales to [0,1]

    img = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1])(img)

    img = img.unsqueeze(0) #add batch
    return img #tensor

def preprocess2_2(image ): #PIL IMAGE
    img = remove_black_borders(image)
    img = img.resize((384,384), resample=0)
    
    img = transforms.PILToTensor()(img)
    
    img = img.float() / 255.0  # Converts to float and scales to [0,1]

    img = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1])(img)

    img = img.unsqueeze(0) #add batch
    return img #tensor

def preprocess2_3(image ): #PIL IMAGE
    img = remove_black_borders(image)
    img = img.resize((800,800), resample=0)
    
    img = transforms.PILToTensor()(img)
    
    img = img.float() / 255.0  # Converts to float and scales to [0,1]

    img = transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1, 1, 1])(img)

    img = img.unsqueeze(0) #add batch
    return img #tensor

def run():
    _show_torch_cuda_info()
    
    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model1
    model1 = model_task1()#init model task 1 
    model1.load()#load weight
    
    model2 = model_task2()
    model2.load()
    
    thresholds = torch.tensor([0.53, 0.52, 0.48, 0.42, 0.46, 0.46, 0.48, 0.4, 0.44, 0.5]).to(device)

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
#         ...

        print(f"Running inference on {jpg_image_file_name}")

        # For example: use Pillow to read the jpg file and convert it to a NumPY array:
        image = Image.open(jpg_image_file_name)
        preprocess_image = preprocess1(image)
        
        #print(type(preprocess_image))
        img = preprocess_image.float().to(device)
        
        output1 = model1.predict(img)
        
        post_output1 = F.softmax(output1, dim=0)
        
        pred = post_output1[1].item()
               
#         is_referable_glaucoma_likelihood = random.random()
        is_referable_glaucoma_likelihood = pred
        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.5
        print(is_referable_glaucoma_likelihood)
        
        if is_referable_glaucoma:
            
            preprocess_image1 = preprocess2_1(image)
            preprocess_image2 = preprocess2_2(image)
            preprocess_image3 = preprocess2_3(image)
            
            img1 = preprocess_image1.float().to(device)
            img2 = preprocess_image2.float().to(device)
            img3 = preprocess_image3.float().to(device)
            
            output = model2.predict(img1,img2,img3)
            #print(output) #[tensor([1], device='cuda:0'), tensor([1], device='cuda:0'), tensor([0], device='cuda:0'), tensor([0], device='cuda:0'), tensor([0], device='cuda:0'), tensor([0], device='cuda:0'), tensor([0], device='cuda:0'), tensor([0], device='cuda:0'), tensor([1], device='cuda:0'), tensor([1], device='cuda:0')]
            #print(output)
            # Convert tensor outputs to boolean values (True for 1, False for 0)
            output_probs = (output.sigmoid() > thresholds).cpu()
            output_probs = output_probs.flatten()
            print(output_probs) #tensor([[ True,  True, False, False, False, False,  True, False, False,  True]])
            
#             features = {
#                 k: v for k, v in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), output_probs)
#             }
            # Création du dictionnaire de caractéristiques
            features = {key: val.item() for key, val in zip(DEFAULT_GLAUCOMATOUS_FEATURES.keys(), output_probs)}

            
            print(features)
        else:
            features = None
#         ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
        
        
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
