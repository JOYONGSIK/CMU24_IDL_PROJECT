import torch
from torch.utils.data import Dataset

class MultimodalSarcasmDataset(Dataset):
    def __init__(self, dlog_ids, utterance_input_ids, utterance_attention_mask, acoustic_data, visual_data, labels, audio_dir, frames_dir, max_frames, frame_size, num_audio_tokens):
        self.dlog_ids = dlog_ids
        self.utterance_input_ids = utterance_input_ids
        self.utterance_attention_mask = utterance_attention_mask
        self.acoustic_data = acoustic_data
        self.visual_data = visual_data
        self.labels = labels
        self.audio_dir = audio_dir
        self.frames_dir = frames_dir
        self.max_frames = max_frames
        self.frame_size = frame_size
        self.num_audio_tokens = num_audio_tokens

    def __len__(self):
        return len(self.dlog_ids)

    def __getitem__(self, idx):
        file_id = self.dlog_ids[idx]
        input_ids = self.utterance_input_ids[idx]
        attention_mask = self.utterance_attention_mask[idx]
        acoustic_input = self.acoustic_data[idx]
        visual_input = self.visual_data[idx]
        label = self.labels[idx]
        # Audio & Visual Data
        audio_path = os.path.join(self.audio_dir, f"{file_id}.wav")
        frames_path = os.path.join(self.frames_dir, file_id)
        return input_ids, attention_mask, acoustic_input, visual_input, label