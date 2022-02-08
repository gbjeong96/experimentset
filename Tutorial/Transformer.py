import torch
from torchaudio import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transforms = ToTensor(), #ToTensor는 image나 ndarray를 FloatTensor로 변환한다.
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y), value=1)) #scatter를 호출하여 10짜리 tensor에 y에 해당하는 index값에 1을 할당하여 원핫 벡터로 만들어준다.
)

#Quick Start에서는 label을 원핫 벡터로 안만들어 줬는데 그 차이가 무엇인지 check