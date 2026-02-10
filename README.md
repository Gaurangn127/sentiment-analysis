# DL_t32025: Deep Learning Project for Sentiment Analysis

This repository contains the code, data, and notebooks for a Deep Learning project focused on Sentiment Analysis, exploring various models like BERT, DeBERTa, and custom LSTMs. The project uses a structured workflow, separating data handling, model training, and inference into distinct script stages.

## Setup and Installation

The environment is set up using the built-in Python Virtual Environment (venv) and the requirements.txt file, ensuring all project dependencies are isolated.

### Prerequisites

- Python 3.9+ (Ensure a recent version is installed).

- A compatible NVIDIA GPU is highly recommended for running the GPU-enabled packages (e.g., PyTorch and TensorFlow).

- System Dependencies: If you are using GPU-enabled packages (like torch==2.5.1+cu121), you must manually ensure the correct version of CUDA and supporting libraries are installed on your host system before running the installation command.

### Environment Setup

1. Navigate to the root project directory:
```bash
cd DL_t32025
```

2. Create the Python Virtual Environment (we'll name it .venv):
```bash
python -m venv .venv
```

3. Activate the environment:

    - macOS/Linux
    ```bash
    source .venv/bin/activate
    ```

    - Windows (PowerShell)
    ```bash
    .venv\Scripts\Activate.ps1
    ```


4. Install all project dependencies using pip:
```bash
pip install -r requirements.txt
```
## Project Structure:
```
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── notebooks/
│   ├── milestones/
│   │   ├── milestone_1.ipynb
│   │   ├── milestone_2.ipynb
│   │   ├── milestone_3.ipynb
│   │   ├── milestone_4.ipynb
│   │   └── milestone_5.ipynb
│   └── models/
│       ├── final-bert-base-uncased.ipynb
│       ├── final-custom-lstm.ipynb
│       └── final-deberta-absa.ipynb    
├── scripts/
│   ├── 01_preprocessing.py
│   ├── 02_training.py
│   └── 03_inference.py
├── README.md
└── requirements.txt
```


## Usage

The project workflow is executed using three sequential scripts located in the `scripts/` directory.

1. **Data Preprocessing**

Run the script to clean, tokenize, and prepare the raw data for model consumption.
```
python scripts/01_preprocessing.py
```

(This script processes raw data from the `data/` directory and saves intermediate artifacts.)

2. **Model Training**

Execute the training script. This will load the processed data and train the Deep Learning model.
```
python scripts/02_training.py
```

(This script saves the trained model weights and checkpoints.)

3. **Inference and Submission**

Use the inference script to load a trained model, generate predictions on the test data (`data/test.csv`), and create a submission file.
```
python scripts/03_inference.py
```

(This script is expected to output a submission file like `submission.csv`.)

## Model Development

The `notebooks/models/` directory contains the detailed experimental analysis and finalization of the deep learning architectures. Key models explored include:

- BERT (`final-bert-base-uncased.ipynb`): Implementation and tuning of the BERT model.

- DeBERTa (`final-deberta-absa.ipynb`): Implementation and tuning of the DeBERTa model, specifically optimized for Aspect-Based Sentiment Analysis (ABSA).

- Custom LSTM (`final-custom-lstm.ipynb`): Development of a custom Long Short-Term Memory (LSTM) network baseline.