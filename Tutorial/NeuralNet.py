import os
import torch
from torch import logit, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) #dim=1은 가로방향 적용, dim=0은 세로방향 적용
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")



#모델 계층(Layer)
#28*28크기의 이미지 3개로 구성된 미니배치
input_image = torch.rand(3,28,28)
print(input_image)

#nn.Flatten
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

#nn.Linear
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

#nn.ReLU
print(f"Before ReLU: {hidden1}\n\n")
# ReLU = nn.ReLU()
# hidden1 = ReLU(hidden1)
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

#nn.Sequential
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

#nn.Softmax
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)



#모델 매개변수
#신명망 내부에 계층들이 parameterize된다. pramaters() 및 named_parameters() 메소드로 모든 매개변수에 접근할 수 있다.
print("Model Structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")