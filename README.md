# Handwritten Digit Recognition on Persian(Farsi) MNIST Dataset

In this project i implemented MLP network using pytorch and train it on Perian MNIST dataset.

## :hammer: Libraries used 

- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)

## :chart_with_upwards_trend: Dataset
I uses persian digit handwritten dataset like [MNIST](http://yann.lecun.com/exdb/mnist/). You can find [here](https://github.com/rezaAdinepour/Persian-Handwritten-Digit-Recognition/tree/main/bmp)

## :key: Neural Network
MLP network is used for this implementation. find MLP class in this file: <code>mlp.py</code>

## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/chandrikadeb7/Face-Mask-Detection.git
```

2. Change your directory to the cloned repo 
```
$ cd Face-Mask-Detection
```

3. Create a Python virtual environment named 'test' and activate it
```
$ virtualenv test
```
```
$ source test/bin/activate
```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
```

2. To detect face masks in an image type the following command: 
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in real-time video streams type the following command:
```
$ python3 detect_mask_video.py 
```
## :key: Results
