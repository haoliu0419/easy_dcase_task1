import os
from model.lit_asc import LitAcousticSceneClassificationSystem
from model.backbones import TFSepNet
from util.spec_extractor import CpMel
from data.data_module import DCASEDataModule
from lightning.pytorch import Trainer
import time
import librosa
import torch

ckpt_file = os.path.join(
    os.getcwd(),
    "log/tfsepnet_train/version_82/checkpoints/epoch=56-val_acc=0.9951.ckpt",
)
model = LitAcousticSceneClassificationSystem.load_from_checkpoint(
    ckpt_file, backbone=TFSepNet(), spec_extractor=CpMel(n_mels=512)
)
model.eval()

wav, sr = librosa.load(
    "../content_detection/test/park-Spanish-podcast-191.wav", sr=32000
)
wav = torch.tensor(wav).unsqueeze(0)
cuda0 = torch.device("cuda:0")
wav = wav.to(cuda0)
model_process_data = CpMel(n_mels=512).to(cuda0)

spec = model_process_data(wav).unsqueeze(0)
y_hat = model(spec)

print(y_hat)
