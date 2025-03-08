# Self-Driving Car - Behavioral Cloning

## Overview
This project implements a **behavioral cloning** approach to train a deep learning model to autonomously drive a car in a simulated environment. The model is trained using a convolutional neural network (CNN) based on the VGG16 architecture, leveraging Udacity's self-driving car simulator dataset. The trained model predicts steering angles based on input images from the car's front-facing camera.

## Dataset
The dataset is collected from the **Udacity Self-Driving Car Simulator** and consists of:
- Center, Left, and Right camera images
- Corresponding steering angles
- Additional telemetry data (throttle, brake, speed)

### Data Augmentation
To improve generalization and prevent overfitting, the dataset is augmented using:
- **Flipping images** (mirroring to simulate opposite turns)
- **Brightness adjustment** (to simulate different lighting conditions)
- **Adding random shadows**
- **Cropping images** (to focus on the road and remove unnecessary background)
- **Gaussian noise and blurring**

## Model Architecture
The model is based on **VGG16** with modifications:
- Removed fully connected layers and replaced them with custom layers for regression
- Used **ELU activation function** for better gradient flow
- Added **Dropout layers** to mitigate overfitting
- Implemented **Batch Normalization** to speed up training

### Model Summary:
1. **Input layer** - 3-channel image (66x200x3)
2. **Convolutional layers** (feature extraction, strides instead of pooling)
3. **Fully connected layers** (to map extracted features to steering angles)
4. **Output layer** - Single neuron predicting the steering angle

## Training and Validation
- **Training Samples**: 3,511
- **Validation Samples**: 878
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (adaptive learning rate)
- **Batch Size**: 32
- **Epochs**: Tuned based on early stopping
- **K-Fold Cross-Validation**: Implemented to improve model robustness

## Performance Metrics
- **Loss Curve Analysis** to monitor overfitting
- **Confusion Matrix & ROC Curve** for model evaluation
- **Visualizing Predicted vs. Actual Steering Angles**

## Deployment
The trained model is deployed in the **Udacity Self-Driving Car Simulator**, where it autonomously drives the car using the predicted steering angles in real-time.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/self-driving-car.git
   cd self-driving-car
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Test the model in the simulator:
   ```bash
   python drive.py model.h5
   ```

## Future Improvements
- Implement **Reinforcement Learning** for better adaptability
- Fine-tune **CNN layers** for improved accuracy
- Test with real-world driving datasets

## Contributors
- **Archit Jain** (jainarchit088@gmail.com)

## License
This project is licensed under the MIT License.

