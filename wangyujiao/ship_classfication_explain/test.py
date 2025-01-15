import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image

classes = ('DDT', 'HC', 'HM', 'QT', 'QZJ', 'YC')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("pretrainedModel.pth")
model.eval()
model.to(DEVICE)
path = 'data/test/YC/1065.jpg'

img = Image.open(path)
img = transform_test(img)
img.unsqueeze_(0)
img = Variable(img).to(DEVICE)
out = model(img)
# Predict
score, pred = torch.max(out.data, 1)
print('Image Name:{},predict:{},score:{}'.format(path, classes[pred.data.item()], score.data.item()))
