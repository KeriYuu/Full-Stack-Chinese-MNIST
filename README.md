# Full-Stack-Chinese-MNIST
This repository contains a full-stack deep learning project focused on Chinese MNIST handwritten digit recognition. The project utilizes the Chinese MNIST dataset, performs data processing, and implements a Convolutional Neural Network (CNN) model for training. The trained model is then integrated into a frontend application and a backend system deployed in Docker. 

## Contents

* `requirements/` - dependencies
* `train.ipynb` - notebook for model training
* `deliver.ipynb` - notebook for model deployment
* `data/` - raw and processed data
* `frontend/` - front-end code
* `backend/` - backend code
* `text_recognizer/` - model deployment code
* `training/` - training code

## Workflow

The workflow of the project is broken down into two main parts: training the model and delivering the model. 

### Training

This part is handled in `train.ipynb` and it consists of the following steps:

1. **Downloading the Chinese MNIST dataset**: The dataset is obtained from Kaggle and downloaded into the project for further processing. 

2. **Data processing**: After downloading, the data is processed and prepared for the training process.

3. **Model training**: A CNN model is trained on the processed data. This training happens both offline and online. The online training saves the trained model (artifact) in Weights & Biases (wandb).

### Delivery

The delivery of the model to the front-end and backend systems is covered in `deliver.ipynb` and it involves:

1. **Checkpoint conversion**: The saved model from wandb is converted into a TorchScript for better compatibility and efficiency.

2. **Frontend build**: The deep learning model is integrated into the frontend system using Gradio. This allows the model to be accessed either locally or from another machine, providing a user-friendly interface for model interaction.

3. **Backend build**: The backend system is set up using AWS Lambda, which serve predictions only when a request hits. The model and its associated application are containerized and managed using Docker, ensuring a smooth migration and deployment process.

## References

The project was developed based on this [repository](https://github.com/the-full-stack/fsdl-text-recognizer-2022/tree/main).

The Chinese MNIST data used in this project can be found on [Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist).

## Installation and Usage

Before running the notebooks, please install the required Python packages by running:

```
pip install -r requirements/prod.txt
```

To train the model, run the `train.ipynb` notebook.

For model deployment, follow the steps provided in `deliver.ipynb`.
