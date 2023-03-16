# Garage Convolutional Neural Network (CNN)

I have a remote garage door WiFi device that is supposed to tell me when the garage door is open or closed. 
However, I found it to be unreliable.  Thus, I am using my security camera and a CNN to tell me if the door is open/closed.

Extra benefit - I can also send myself alerts if the garage door is left open for n number of minutes.

## Classes
I have two vehicles in the garage.  Therefore, there are 4 vehicle states:
* both cars
* no cars
* one car
* other car

Plus, each of those states has two more states, whether the garage door is open or closed.  This brings the total to 8 states (or classes).

## Categorizing Images for Training
I had the security camera take a picture every minute in the garage for a week.  Then, I sorted them with the script 'categorize.py'.  The script goes minute by minute, using the library skimage.metrics.structural_similarity to compare the two most recent images.  

If there is a small difference between the two images, it continues and gathers all images it passes into an array.  

Once it detects a big change (car or garage moved), it prompts me to categorize the gathered images and moves them to a folder for PyTorch's later use.

## Training
### Preprocessing
In order to make the model resilient, all images are a preprocessed.  The garage windows are cropped out of the top of the image. The date and time are cropped out the bottom of the image (from the security camera), and each image is resized and grayscaled.

### Sample Size
I have far more photos of both vehicles in the garage with the door closed than I do any other state.  To compensate, I made sure to take an even number of samples from each dataset.  I randomly pick an image, then apply random center-cropping and rotation, and feed that to PyTorch.

### Building the Model
The model is very simple. Please see ModelClass in common.py. 

### GPU
If an Nvidia GPU is available with Cuda installed, Pytorch will switch to that device for training.  Else, it will default to CPU.