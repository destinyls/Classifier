import argparse
import os
import json
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm
from model.model import initialize_model, save_checkpoint
from data.person_dataloader import build_dataloader

parser = argparse.ArgumentParser(description='Begin Inference ...')
parser.add_argument('--checkpoints_path', type=str, default="./", help='the path of checkpoints')
parser.add_argument('--submit_results_path', type=str, default="./result.json", help='the path of submit files')
parser.add_argument('--model_name', type=str, default='resnet', help= 'the name of model')

args = parser.parse_args()
checkpoints_path = args.checkpoints_path
submit_results_path = args.submit_results_path
model_name = args.model_name

def test_fn(model, dataloader):
    results = []
    for image in tqdm(dataloader):
        image = image.to(device)
        with torch.set_grad_enabled(False):
            if is_inception:
                outputs, aux_outputs = model(image)
            else:
                outputs = model(image)
            outputs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            results.append(outputs)
    results = np.vstack(results)
    return results

if __name__ == "__main__":
    data_root =  "dataset"
    num_classes = 3
    is_inception = (model_name == "inception")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model
    model, _ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True)
    model = model.to(device)
    checkpoints = torch.load(checkpoints_path)
    model.load_state_dict(checkpoints["model_state_dict"])
    model.eval()
    _, _, testloader = build_dataloader(data_root, batch_size=1)
    results = test_fn(model, testloader)
    
    for i in range(9):
        print("----------crop----------： ", i)
        results += test_fn(model, testloader)
    results /= 10
    results_label = np.argmax(results, axis=1)
    results_score = np.max(results, axis=1).tolist()
    
    results_json = []
    class_dict = {0: "smoking", 1: "calling", 2: "normal"}
    image_path = pd.read_csv(os.path.join(data_root, 'test.csv'))["image_path"]
    for i in range(image_path.shape[0]):
        image_name = image_path[i].split('/')[1]
        results_json.append({"image_name": image_name, "category": class_dict[results_label[i]], "score": results_score[i]})

    with open(submit_results_path,'w') as json_file:
        json.dump(results_json, json_file)