import torch

from models import RegNet, model_dict


def load_checkpoint(checkpoint_file: str, num_classes: int = 62)\
        -> (RegNet, int, float, float):
    with open(checkpoint_file, 'rb') as fp:
        checkpoint = torch.load(fp, map_location='cpu')

    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    val_acc = checkpoint['val_acc']
    model_state_dict = checkpoint['model_state']
    optimizer_state_dict = checkpoint['optimizer_state']
    model_type = checkpoint['config']['MAIN']['model']

    model = RegNet(num_classes, model_dict[model_type], 0,  pretrained=False)
    model.load_state_dict(model_state_dict)

    return model, epoch, val_loss, val_acc


