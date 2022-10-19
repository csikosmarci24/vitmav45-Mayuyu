import torch


if torch.cuda.is_available():
	print("Hello GPU")
else:
	print("Hello CPU")