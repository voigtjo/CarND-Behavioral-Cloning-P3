# **Traffic Sign Recognition** 

### Writeup for the CarND-Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[cnn_architecture]: ./images/cnn-architecture.png "CNN Architecture"
[model_lost]: ./images/model_lost_8.png "Model Lost"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code Using the Udacity provided simulator and my drive.py file; the car can be driven autonomously around the track by executing

```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. I applied the model from nvidia for autonomous driving https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

![alt text][cnn_architecture]  

####2. Attempts to reduce overfitting in the model

I didn't modify the model by applying regularization techniques like [Dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) or [Max pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Max_pooling_shape). 
I played around with the number of the training epochs low and kept a very with just two epochs very low.

####3. Model parameter tuning

The model used an adam optimizer with Mean Squared Error, with the default learning rate from 0.001. 


####4. Appropriate training data

I imported the images of the center, left and right cameras of the provided training data. The angle for the left and right camera was corrected by +0.25 and -0.25. The test data has been extended by flipped images since the training data describe a large counterclockwise curve.

The system was trained on 35872 samples and validated on 9644 samples. The following picture shows the training for 8 epochs and the overfitting of the data. The validation loss doesn't decrease after the second epoch.

![alt text][model_lost]

After the training with two epochs (or alternativly more), the car was driving down the road all the time on the first track: video.mp4