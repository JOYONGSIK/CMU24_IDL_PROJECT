import os
import torch
import torchaudio
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype, Resize, Normalize
from PIL import Image


def pad_seq(tensor, dim, max_len):
    """Sequence Padding."""
    if max_len > tensor.shape[0]:
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    return tensor[:max_len]

def load_audio(audio_path, num_audio_tokens):
    if not os.path.exists(audio_path):
        return torch.zeros(1, num_audio_tokens)  

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform[0]  
    clip_duration = len(waveform) / sample_rate
    if clip_duration > 0:
        new_sample_rate = min(int(sample_rate * num_audio_tokens / clip_duration), 16000)
        waveform = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
    return waveform.unsqueeze(0)  

def load_frames(frames_path, max_frames, frame_size):
    if not os.path.exists(frames_path):
        return torch.zeros(max_frames, 3, frame_size, frame_size)

    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    target_frames = torch.linspace(0, len(frame_files) - 1, max_frames).long()
    transforms = Compose([
        ToTensor(),
        ConvertImageDtype(torch.float32),
        Resize(frame_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = [transforms(Image.open(os.path.join(frames_path, frame_files[i]))) for i in target_frames]
    return torch.stack(frames)