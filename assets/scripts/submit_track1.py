from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys 
import os
from glob import glob
from torchvision import transforms
import torch
import pandas as pd
import numpy as np

class ImgDset(Dataset):
    def __init__(self, root, transform):
        self.img_fs = sorted(glob(os.path.join(root, "*.png")))
        print(f"Loaded {root} dir with {len(self.img_fs)} entries")
        self.transform = transform

    def __getitem__(self, indx):
        img_f = self.img_fs[indx]
        img = Image.open(img_f).convert("RGB")
        img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_fs)

def create_model(ckpt):
    # DEFINE YOUR MODEL HERE AND CREATE AN OBJECT
    """
    Eg:
    from models import DeepFakeDetector
    model = DeepFakeDetector(**args) 
                (or)
    class DeepFakeDetector(nn.Module):
        def __init__(self, *args):
            .....
            .....
    model = DeepFakeDetector(**args) 
    """
    model = None # TODO: FILL HERE

    # LOAD TRAINED CHECKPOINT
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    return model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Incorrect arguments. Please run as: python3 submit_track1.py <path-to-test-datadir> <model-ckpt>")
    
    # CREATING TEST DSET
    test_datadir = sys.argv[1]
    if not os.path.exists(test_datadir):
        raise ValueError("Incorrect data directory path")
    
    # IMG TRANSFORM. NOTE: YOU ARE NOT ALLOWED TO RESIZE THE IMG
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dset = ImgDset(test_datadir, transform)
    # feel free to change the batch size and num of workers based on available compute
    test_loader = DataLoader(test_dset, batch_size=32, shuffle=False, num_workers=4)

    # DEFINE MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = sys.argv[2]
    if not os.path.exists(model_ckpt):
        raise ValueError("Incorrect model checkpoint path")
    model = create_model(model_ckpt)
    model = model.to(device)

    prob_real = []
    for indx, img in enumerate(test_loader):
        img = img.to(device)
        
        # PASS BATCH OF IMAGES TO MODEL
        # feel free to modify based on need
        pred = model(img)
        # NOTE: ASSUMING NO NORMALIZATION
        # if SOFTMAX (model outputs in the form fake=[1, 0], real=[0, 1])
        if pred.shape[1] == 2:
            pred = torch.softmax(pred, dim=1)
            p_real = pred[:,1]
        # if SIGMOID (model outputs in the form fake=0, real=1)
        elif pred.shape[1] == 1:
            pred = torch.sigmoid(pred)
            p_real = pred[:,0]

        prob_real.extend(p_real.detach().cpu().numpy())

        print(f"[{indx}/{len(test_loader)}]", end="\r")
    prob_real = np.array(prob_real)
    # clip predicted probabilties to avoid infinite error
    prob_real = np.clip(prob_real, 0.025, 0.975)

    print("Done... Creating submission file")

    df = pd.DataFrame()
    df["id"] = np.arange(len(prob_real))
    df["p_real"] = prob_real
    df.to_csv("submission.csv", index=False)
    print("Submission file created at ./submission.csv")