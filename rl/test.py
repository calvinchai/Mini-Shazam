from retrieval import wav_to_rep
from models import SpecAutoEncoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model
model = SpecAutoEncoder()
model_name = "SpecAutoEncoder"

model.load_state_dict(torch.load("../models" + f"/{model_name}_best.pt"))
model.to(device)

wav_file_path = r"E:\cs682\data\fma_small_wav\000002.wav"
rep = wav_to_rep(wav_file_path, model, device)
print(rep)