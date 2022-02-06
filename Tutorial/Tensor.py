import torch
import numpy as np

#데이터로부터 직접 directly 생성
data = [[1,2],[3,4]]
x_data = torch.tensor(data)


#NumpPy 배열로부터 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)


#다른 텐서로부터 정의하기
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float ) # x_data의 속성을 덮어쓴다.
print(f"Random Tensor: \n {x_rand} \n")


#무작위(random) 또는 상수(constant) 값을 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


#텐서의 속성(Attribute)
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


#텐서 연산(Opearation)
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

#Numpy식의 표준 인덱싱과 슬라이싱:
tensor = torch.ones(4,4)
print('Frist row: ', tensor[0])
print('Frist column: ', tensor[:,0])
print('Last colunm: ', tensor[...,-1])
tensor[:,1] = 0

#텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 요소별 곱(element-wise product)을 계산합니다. z1, z2, z3는 모두 같은 값을 갖습니다.
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor, out=z3)

#aggregate
agg = tensor.sum()
agg_item = agg.item() #item은 scalar 값만 가져올수있음. (tensor의 element를 가져온다. scalar 값만)
print(agg_item, type(agg_item))

#바꿔치기(in-place) 접미사 _를 사용한다. ex) x.copy_(y), x.t_()
print(tensor, "\n")
tensor.add_(5)
print(tensor) #in-place 연산은 메모리를 절약하지만, 기록(history)가 즉시 삭제되어 도함수 계산에 문제가 발생할 여지를 줌. 사용 권장 x


#Numpy 변환(Bridge)
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1) #cpu상의 tensor와 numpy는 메모리를 공유하기 때문에 하나만 바꿔도 둘다 바뀜.
print(f"t: {t}")
print(f"n: {n}")


#Numpy배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")