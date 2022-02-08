from turtle import down, forward
import torch
from torch import flatten, nn
from torch.utils.data import DataLoader
from torchvision import datasets #CIFAR,COCO...
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

print(":: Dataset allocate ::")

batch_size = 64

#Data Loader Create -> Dataste을 순환 가능한 객체로 만든다.

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size= batch_size)

# print("======Iteration======")
# for X, y in test_dataloader:
#    print("Sahpe of X [N,C,H,W]: ", X.shape)
#    print("Shape of y: ", y.shape, y.dtype)
# print("======End======")

#학습에 필요한 CPU, GPU 장치를 얻는다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#모델 정의하기
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)


#모델 매개변수 최적화 하기
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)
        
        #역전파
        optimizer.zero_grad() #모델 매개변수의 변화도를 재설정, 기본적으로 변화도는 더해지기(add up) 때문에, 중복계산을 막기 위해서 반복할때마다 명시적으로 0으로 설정해야한다.
        loss.backward() #loss에 대한 requires_grad가 True인 변수의 변화량 값 계산.
        optimizer.step() #변화도를 계산한 뒤에 해당 함수를 호출하면 수집된 변화도로 매개변수를 조정한다.
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batchs = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad(): #연산 중에서 gradient를 계산하는 기록 추적, 변화도 계산 지원이 필요 없는 경우 사용 (연산추적 중지)
        for X, y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #배치 안에 정답인 거 전부다 sum
    test_loss /= num_batchs
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epoch = 10
for t in range(epoch):
    print(f"Epoch {t+1}\n------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#모델 저장하기
torch.save(model.state_dict(), "model.pth")
print("Saved Pytorch Model State to model.pth")

#모델 불러오기
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))


#모델을 이용하여 예측
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')