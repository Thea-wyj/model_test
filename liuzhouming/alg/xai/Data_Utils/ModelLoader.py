import torch

def load_model_pt(model_url, save_path):
    path_split = model_url.split('/')
    filepath = save_path+"/download/model/" + path_split[-1]
    return torch.load(filepath)
