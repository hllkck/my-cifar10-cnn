# my-cifar10-cnn

Project Steps
1. Importing Required Libraries
   I imported the necessary libraries for data processing, model building, training and visualization.
2. Loading the Dataset
   I have combined all the file paths into a single file path for the CIFAR-10 dataset.
3. Combining Data Batches
   Each batch is loaded, and image data and their corresponding labels are extracted. These are combined into a single dataset.
4. Data Preprocessing
   The image data X is reshaped into a format suitable for CNNs: 32x32 pixels with 3 channels (RGB).
   Pixel values are normalized to the range [0, 1].
   The class labels y are one-hot encoded.
5. Splitting the Dataset
   The dataset is split into training and testing sets. 80% of the data is used for training, and 20% is used for testing.
6. Defining the CNN Model
   Convolutional Layers: Extract features from the input images.
   MaxPooling Layers: Reduce the spatial dimensions of feature maps.
   Dense Layers: Classify the extracted features.
   Dropout: Prevent overfitting.
   Output Layer: Contains 10 neurons (one for each class), using a softmax activation function.
7. Compiling the Model
   The model is compiled using:
    Optimizer: Adam, for efficient gradient-based optimization.
    Loss Function: Categorical Crossentropy, for multi-class classification.
    Metrics: Accuracy, to track model performance during training.
8. Training the Model
   Callbacks:
    EarlyStopping: Stops training when validation loss stops improving.
    ReduceLROnPlateau: Reduces the learning rate when validation loss plateaus.
   Parameters:
    epochs: Maximum number of training iterations.
    batch_size: Number of samples processed together in a single pass.
    validation_data: The testing set is used for validation during training.
9. Visualizing Training Metrics
    The loss and accuracy for both training and validation datasets are plotted to analyze model performance over epochs.
10. Evaluating the Model
    The model is evaluated on the testing set, and the test loss and accuracy are printed.
11. Making Predictions
    The model’s predictions on the test set are displayed for a few examples to compare with actual labels.

Results
The trained CNN model achieves approximately 87% accuracy and 67% test accuracy (depending on execution). 
Further tuning and augmentation can improve the results.

Visual outputs of the training process are as follows:




