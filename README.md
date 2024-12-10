# Multi-Modal Sarcasm Detection Using Cognitive Load

This project implements a multimodal sarcasm detection model using audio, visual, and text data. The model leverages the BART architecture with additional cognitive load features for enhanced accuracy.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JOYONGSIK/CMU24_IDL_PROJECT
   cd multimodal-sarcasm-detection
   ``` 

2.	Install the required packages: 
```bash 
pip install -r requirements.txt
```

3.	Download the pre-trained models and place them in the ./models directory.

##### Project Structure 

```bash 
.
├── data_utils.py       # Data loading and preprocessing utilities
├── dataset.py          # Dataset class for multimodal sarcasm detection
├── models/
│   ├── cognitive_load.py  # Cognitive Load feature extractor
│   ├── fusion_modules.py  # Fusion modules for cross-modal attention
│   ├── multimodal_bart.py # Multimodal BART model definition
│   └── attention_modules.py # Attention modules
├── train.py            # Training loop
├── test.py             # Testing loop
├── main.py             # Main script for running the pipeline
├── requirements.txt    # List of required Python packages
├── README.md           # Project documentation
└── saved_models/       # Directory for saving trained models
```

###  Usage
To train the model, run the following command: 
```bash 
python main.py
```

### Testing

After training, the model will be automatically tested on the test set.

### Data Preparation

Prepare your datasets (train.pkl, valid.pkl, test.pkl) and place them in the ./data directory. Ensure the data is preprocessed correctly.

### Model Details 

This project uses a multimodal BART architecture, which includes:
- Text Input: Tokenized textual data using BART tokenizer.
- Audio Input: Processed waveform data using Wav2Vec2.
- Visual Input: Image frames processed using ResNet.
- Cognitive Features: Extracted from pre-trained cognitive load models.
