# Driver Distraction Detection Using Raspberry Pi 

## Motivation

In recent years, the increasing number of vehicles on roads leads to
an increase in traffic accidents. In 2015, the National Highway
Traffic Safety Administration, part of U.S. Department of
Transportation, reported that 35,092 people died in traffic accidents
on the U.S. roads, a 7.2% increase in fatalities from 2014.

Distracted driving was responsible for 391,000 injuries and 3477
fatalities in 2015 .

It is found that distracted driving was related
to one-tenth of fatal crashes. Distracted driving fatalities have
increased more rapidly than those caused by drunk driving,
speeding and failing to wear a seatbelt.

A driver is considered to be distracted when there is an activity
that attracts his/her attention away from the task of driving. There
are three types of driving distractions:

• Manual distraction: The driver takes his/her hands off the
wheel, e.g. drinking, eating etc.

• Visual distraction: The driver looks away from the road, e.g.
reading, watching the phone etc.

• Cognitive distraction: The driver's mind is not

This project propose a deep learning architecture to overcome this major issue using CNN based driver distraction system working in real time which can be deployed on raspberry pi. 

## Installation

## Section 1 - How to Set Up and Run TensorFlow Lite Object Detection Models on the Raspberry Pi

Setting up TensorFlow Lite on the Raspberry Pi is much easier than regular TensorFlow! These are the steps needed to set up TensorFlow Lite:

- 1a. Update the Raspberry Pi
- 1b. Download this repository and create virtual environment
- 1c. Install TensorFlow and OpenCV
- 1d. Set up TensorFlow Lite detection model
- 1e. Run TensorFlow Lite model!

### Step 1a. Update the Raspberry Pi
First, the Raspberry Pi needs to be fully updated. Open a terminal and issue:
```
sudo apt-get update
sudo apt-get dist-upgrade
```
Depending on how long it’s been since you’ve updated your Pi, the update could take anywhere between a minute and an hour. 

While we're at it, let's make sure the camera interface is enabled in the Raspberry Pi Configuration menu. Click the Pi icon in the top left corner of the screen, select Preferences -> Raspberry Pi Configuration, and go to the Interfaces tab and verify Camera is set to Enabled. If it isn't, enable it now, and reboot the Raspberry Pi.



### Step 1b. Download this repository and create virtual environment

Next, clone this GitHub repository by issuing the following command. The repository contains the scripts we'll use to run TensorFlow Lite, as well as a shell script that will make installing everything easier. Issue:

```
git clone https://github.com/himasha0421/Driver-Distraction-Detection-With-Pi.git
```

This downloads everything into a folder called TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi. That's a little long to work with, so rename the folder to "tflite1" and then cd into it:

```
mv Driver-Distraction-Detection-With-Pi tflite1
cd tflite1
```

We'll work in this /home/pi/tflite1 directory for the rest of the guide. Next up is to create a virtual environment called "tflite1-env".

I'm using a virtual environment for this guide because it prevents any conflicts between versions of package libraries that may already be installed on your Pi. Keeping it installed in its own environment allows us to avoid this problem. For example, if you've already installed TensorFlow v1.8 on the Pi, you can leave that installation as-is without having to worry about overriding it.

Install virtualenv by issuing:

```
sudo pip3 install virtualenv
```

Then, create the "tflite1-env" virtual environment by issuing:

```
python3 -m venv tflite1-env
```

This will create a folder called tflite1-env inside the tflite1 directory. The tflite1-env folder will hold all the package libraries for this environment. Next, activate the environment by issuing:

```
source tflite1-env/bin/activate
```

**You'll need to issue the `source tflite1-env/bin/activate` command from inside the /home/pi/tflite1 directory to reactivate the environment every time you open a new terminal window. You can tell when the environment is active by checking if (tflite1-env) appears before the path in your command prompt, as shown in the screenshot below.**

At this point, here's what your tflite1 directory should look like if you issue `ls`.

<p align="center">
  <img src="/doc/tflite1_folder.png">
</p>

If your directory looks good, it's time to move on to Step 1c!

### Step 1c. Install TensorFlow Lite dependencies and OpenCV
Next, we'll install TensorFlow, OpenCV, and all the dependencies needed for both packages. OpenCV is not needed to run TensorFlow Lite, but the object detection scripts in this repository use it to grab images and draw detection results on them.

To make things easier, I wrote a shell script that will automatically download and install all the packages and dependencies. Run it by issuing:

```
bash get_pi_requirements.sh
```

This downloads about 400MB worth of installation files, so it will take a while. Go grab a cup of coffee while it's working! If you'd like to see everything that gets installed, simply open get_pi_dependencies.sh to view the list of packages.

**NOTE: If you get an error while running the `bash get_pi_requirements.sh` command, it's likely because your internet connection timed out, or because the downloaded package data was corrupted. If you get an error, try re-running the command a few more times.**

**ANOTHER NOTE: The shell script automatically installs the latest version of TensorFlow. If you'd like to install a specific version, issue `pip3 install tensorflow==X.XX` (where X.XX is replaced with the version you want to install) after running the script. This will override the existing installation with the specified version.**

That was easy! On to the next step.


## Usage

```python
1. notebooks/driver_distraction_experiment.ipynb -- experiments with different cnn architectures
2. driver-distraction_mymodel.ipynb - my cnn model 
```
* To run the application follow below command
```python
source tflite1-env/bin/activate
python TFLite_detection_video.py
```

* Download the dataset [kaggle dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection)


go through above notebook to understand how CNN model can be used in this domain . 

## Results

* Results from solved environment ![](doc/model_results.png) .

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Apache License 2.0