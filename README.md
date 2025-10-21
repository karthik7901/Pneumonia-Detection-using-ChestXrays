# Pneumonia Detection Using Chest X-rays
This project applies Deep Learning techniques to classify chest X-rays as Pneumonia or Normal.The repository includes training notebooks for both models, pretrained weights, and a testing notebook for easy testing on sample images.

## Data Preparation
The dataset used was the NIH Chest X-ray Dataset from Kaggle. We divided the dataset into training, validation, and test sets.

Each subset contains two folders:

    /NORMAL — images of healthy lungs

    /PNEUMONIA — images showing signs of infection

Data preprocessing included:

    Resizing images to (224 × 224)

    Normalizing pixel values

    Applying data augmentation for better generalization

## Model Development
We implemented two deep learning models using transfer learning with TensorFlow/Keras:
### VGG16 Model
    Imported the VGG16 model with pre-trained ImageNet weights.
    
    Removed the top layers.
    
    Added custom dense layers for binary classification.
    
    Frozen the convolutional base for initial epochs, then fine-tuned.
    
    Saved trained model weights as: Model Weights/best_vgg16.pth

### ResNet50 Model
    Implemented similarly using the ResNet50 architecture.
    
    Utilized skip connections that help avoid vanishing gradient problems.
    
    Achieved slightly better validation accuracy compared to VGG16.
    
    Saved trained model weights as: Model Weights/best_resnet50.pth

We have used the help of OPENAI(ChatGPT) for some insights.

## Model Training
Both models were trained using:

    Binary Cross-Entropy Loss

    Adam Optimizer

    Batch size: 32

    These trained model weights are stored under model_weights for reuse and make real-time predictions.

During training, we monitored:

    Training vs. Validation Accuracy

    Training vs. Validation Loss

    Confusion Matrix
    
    ROC Curve
All these metrics are saved in VGG Metrics and ResNet Metrics accordingly.

## Evaluation
Each model was evaluated on the test dataset to measure real-world performance. We calculated:

    Accuracy
    
    Precision
    
    Recall
    
    F1-score
    
    ROC-AUC

## Model Testing
To demonstrate real-world usability, we built a testing notebook (03_model_testing.ipynb) that:

    Loads trained model weights (.pth files)

    Reads images from the Sample Images/

    Preprocesses and predicts labels

    Displays output with prediction confidence

## Installation and Setup
-> Open the repository in Colab: https://github.com/karthik7901/Pneumonia-Detection-using-ChestXrays

-> Download the dataset and upload it to the drive. Mount the drive to the colab. Give the location of the dataset as the dataset_directory.

-> Open notebooks/Pneumonia_detection_using_chest_Xrays_using_VGG (3).ipynb:

    1.Load and preprocesses the dataset.
    2.Trains the model on training data.
    3.Save trained weights to Model Weights/best_vgg16.pth.
    4.Generate accuracy/loss plots in VGG Metrics folder.
-> Open notebooks/Pneumonia_Detection_using_Chest_Xrays_with_the_help_of_Resnet (2).ipynb:

    1.Use the same dataset.
    2.Train a ResNet50-based model.
    3.Save weights to Model Weights/best_resnet50.pth.
    4.Export results to ResNet Metrics folder.
-> Open notebooks/Pnemonia_Prediction.ipynb:

    1.Load pretrained weights which are saved to the model_weights directory.
    2.Run inference on images in sample_images/.
    3.Display predictions.
## Metrics and Visualization
Training metrics for each model are stored in the metrics/ directory:

1. Accuracy vs. Epochs

2. Loss vs. Epochs

3. Confusion Matrix

4. ROC-AUC Curve

Each notebook automatically generates and saves these plots.
