# Vision based Human Action Recognition

The "object_detection" folder contains all the files required to run this project. There are a few dependencies in the folder "slim".

To run the project, simply navigate to the "object_detection" folder and type:

```
python3 final_project.py
```
And you get the GUI, written in Tkinter.

There are 2 modes of operation:
* Live mode: This works on realtime video from your webcam
* File mode: This opens a file, processes it frame by frame, and saves it to a file called "output.avi"

Dependencies:
* Keras
* Tensorflow

The script used for training is "train_eg.py". To load your own dataset, put images in folders "training" and "testing", and inside those, put the images into "pick" and "idle" based on their category.

The saved model configuration is in "object_detection/model1.json", and the weights are in "object_detection/fifth_model.h5"