import torch

activation_dict ={
    "ReLU": torch.nn.ReLU,
    "LeakyReLU": torch.nn.LeakyReLU,
    "GELU": torch.nn.GELU,
    "Linear": torch.nn.Identity,
}

initialization_dict = {
    "ReLU": 'relu',
    "LeakyReLU": 'relu',
    "GELU": 'relu', # for 'he' intitializatoin
}

optimizer_dict = {
    "AdamW": torch.optim.AdamW,
}

loss_function_dict = {
    "CrossEntropy": torch.nn.CrossEntropyLoss,
    "MSE": torch.nn.MSELoss,
}