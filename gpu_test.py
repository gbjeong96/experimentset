import torch
import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"
print("======================================================")
print("GPU 사용 가능 여부 : ",torch.cuda.is_available())
print("-----------------------------------------------------")
print("0번 GPU 정보 : ", torch.cuda.get_device_name(0))
print("-----------------------------------------------------")
print("사용 가능한 GPU 갯수 : ", torch.cuda.device_count())
print("======================================================")