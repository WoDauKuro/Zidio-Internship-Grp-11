
# LSTM Speech Emotion Recognition

## Overview
This project demonstrates how to build an LSTM (Long Short-Term Memory) model for recognizing emotions from speech. The notebook includes steps for loading audio data, preprocessing, training an LSTM model, and saving checkpoints for the best-performing model during training.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Model Evaluation](#model-evaluation)
9. [Acknowledgments](#acknowledgments)

## Introduction
Emotion recognition from speech is a challenging task that combines aspects of audio signal processing and deep learning. This project uses an LSTM model to classify audio samples based on the emotions they convey. The LSTM model is particularly well-suited for this task due to its ability to handle sequential data and learn temporal dependencies.

## Dataset
The dataset used is the **Toronto Emotional Speech Set (TESS)**, which contains 2800 audio files categorized by different emotions. Each file is labeled with an emotion that it represents.

- **Dataset Source**: [Kaggle TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Emotions**: The dataset includes emotions like happy, sad, angry, neutral, etc.

## Project Structure
The key steps in the project are:
1. **Data Loading**: Extracting audio files and labels.
2. **Preprocessing**: Converting audio files into a format suitable for the LSTM model.
3. **Model Training**: Building and training the LSTM model with checkpoints to save the best model.
4. **Evaluation**: Assessing the model's performance using metrics like confusion matrices and classification reports.

## Prerequisites
- Python 3.7 or higher
- Libraries: NumPy, pandas, matplotlib, seaborn, librosa, scikit-learn, TensorFlow/Keras

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/WoDauKuro/Zidio-Internship-Grp-11.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Zidio-Internship-Grp-11
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   (Make sure to create a `requirements.txt` file with all necessary libraries, including TensorFlow, librosa, pandas, etc.)

## Usage
1. Download the dataset from Kaggle and extract it into the project directory.
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook lstm_speech_emotion_recognition.ipynb
   ```
3. Follow the steps in the notebook to preprocess the data, train the LSTM model, and evaluate the results.

## Results
The notebook provides visualizations such as confusion matrices and classification reports to help understand the model's performance on the test data. These evaluations highlight how well the LSTM model distinguishes between different emotions.

## Model Evaluation
- The LSTM model's performance is measured using accuracy, precision, recall, and F1-score.
- Checkpoints are used to save the model that performs best on the validation set during training.

## Acknowledgments
- The Toronto Emotional Speech Set (TESS) dataset is sourced from Kaggle.
- This project utilizes various Python libraries, including TensorFlow/Keras, for building and training the LSTM model.
