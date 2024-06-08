import yaml
import glob

from PIL import Image
from torchvision import transforms
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader, Dataset

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

train_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), img_path


if __name__ == '__main__':

    dataset = ImageDataset(glob.glob('jpg/*.jpg'), train_transform)
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, num_workers=0)

    model = inception_v3(weights='DEFAULT')
    model.eval()

    print(model)

    # for batch, _ in dataloader:
    #     with torch.no_grad():
    #         out = model(batch)

    #     prob = torch.nn.functional.softmax(out, dim=1)
    #     print(prob.shape)
