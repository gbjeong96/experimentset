import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5,3, requires_grad=True) #w와 b는 최적화 해야하는 매개변수이기 때문에, 손실함수의 변화도를 계산하기 위해서requires_grad 속성을 사용한다.
#w.requires_grad_(True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x,w)+b #연산 그래프에 적용되는 함수는 Function class의 객체이며, 순전파, 역전파에 대한 계산 방법을 알고있음.
                        #역방향 전파 함수에 대한 참조(reference)는 tensor의 grad_fn 속성에 저장됨.
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print("Gradient function for z = ", z.grad_fn)
print("Gradient function for loss = ", loss.grad_fn)

#변화도(Gradient) 계산하기
loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x,w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w)+b
print(z.requires_grad)

#연산추적을 중지하는 다른 방법
z = torch.matmul(x,w)+b
z_det = z.detach()
print(z_det.requires_grad)



#선택적으로 읽기(Optional Reading): 텐서 변화도와 야코비안 곱(Jacobian Product)
inp = torch.eye(5, requires_grad=True) #대각 one 행렬?
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)