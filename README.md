# Cutting Tool Wear Sound Classification

This repository contains code to classify cutting tool wear using sound data. The model has been trained using sound recordings of cutting tools to identify the level of wear. The project leverages pre-trained models such as `tf_efficientnet_b1_ns` and `ast-finetuned-audioset` to fine-tune the classification task. 

## Dataset

The dataset used for this project can be found on Kaggle: [Cutting Tool Wear Audio Dataset](https://www.kaggle.com/datasets/nachiketsoni/cutting-tool-wear-audio-dataset/versions/1). It contains audio recordings of cutting tools in various conditions of wear.

## Models Used

We use two pre-trained models for transfer learning:
1. **EfficientNet**: We use the `tf_efficientnet_b1_ns` model from the EfficientNet family.
2. **Audio Spectrogram Transformer (AST)**: We fine-tune the `ast-finetuned-audioset` model, which is well-suited for audio classification tasks.

## Repository Structure

* `samples`: Directory containing audio files for classification
* `utils`: Utility scripts for data preprocessing and feature extraction
* `weights`: Directory containing pre-trained model weights
* `main.py`: Main script for running the sound classification model
* `README.md`: This file
* `Training.ipynb`: Jupyter Notebook for training the model
* `Model_Converter.ipynb`: Jupyter Notebook for converting the model

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- torchaudio
- timm (for EfficientNet)
- Hugging Face Transformers (for AST model)

Install the required packages:

```bash
pip install torch torchaudio timm transformers
```

## Running Inference

To run inference on an audio file using the pre-trained model, use the following command::
```
python3 main.py --model_path=weights/full_model.pt --audio_path=samples
```
This will load the pre-trained model from `weights/full_model.pt` and classify the audio files in the `samples` directory.


## Training
To train the model from scratch or fine-tune it on your dataset, use the Training.ipynb notebook. Ensure that the dataset is downloaded and placed in the appropriate folder.

## Model Conversion
If you need to convert the model into another format (e.g., TorchScript or ONNX), you can use the Model_Converter.ipynb notebook.

## Results
The model achieves high accuracy in detecting wear levels in cutting tools based on sound recordings. More detailed results and analysis can be found in the Training.ipynb notebook.