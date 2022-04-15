import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets.folder import pil_loader


class Artists(torch.utils.data.Dataset):
    def __init__(self, base_path, image_ids_fn, images_dir, train):
        self.base_path = base_path
        self.image_ids_fn = image_ids_fn
        self.images_dir = images_dir
        with open(base_path + image_ids_fn, 'r') as fp:
            rows = list(fp)
            self.fnames = [s.strip().split(',')[0] for s in rows[1:]]
            self.img_class_ids = [int(s.strip().split(',')[1]) for s in rows[1:]]
            self.img_ids = list(range(len(self.fnames)))
        self.train = train
        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5138, 0.4915, 0.4315], [0.2675, 0.2572, 0.2626])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5138, 0.4915, 0.4315], [0.2675, 0.2572, 0.2626])
            ]),
        }

    def __getitem__(self, index):
        img_fname = self.fnames[index]
        image = pil_loader(self.base_path + self.images_dir + img_fname)

        if self.transform is not None:
            if self.train:
                image = self.transform['train'](image)
            else:
                image = self.transform['val'](image)
        return image, self.img_class_ids[index]

    def __len__(self):
        return len(self.img_ids)
