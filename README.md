# DROIDvision
SDC Project: DROIDvision

name change: augr +1

Current Capability Workflow:

1- RaspiCam takes pictures at regular interval.

2- Each shot is sent through object detector, which spits out a list of detections for that frame.

3- For each detection in a single list of detections, location is estimated based on distance from camera, orientation of Vision Module, and GPS coords of Vision Module. So for each detection in each frame, we have a 2d vector of the form
(CLASS_OF_OBJECT_DETECTED, (LATITUDE, LONGITUDE)).

4- Each 2d vector is plotted with a python library called Folium. 

## Repo Tour

### Main
Main python file is called main_scope.py under main directory. This file still requires quite a bit of cleanup/modularization but here are the important parts:

-The compiled vision model graph file is called ssdMobileNetGraph. This is in turn points to the actual caffemodel itself. But in main_scope.py, we only care about the graph file. This file is instantiated in main().

-All localization stuff is contained in this file (the lowest hanging fruit in terms of necessary cleanup). The first 4 functions are all localization-related.

-As mentioned above, each frame is sent through this whole gauntlet of processing. The source of this frame extraction is in main() -> "for frame in camera.capture_continuous(...)". Track generation will require an overhaul of this section.

### Detection

all models and model construction stuff goes here

convert model

Vision models are contained here. This project currently uses a MobileNet SSD model using the Caffe framework. I originally made this choice because MobileNet Caffe is compatible with the Intel NCS, the vision processing hardware we are currently using. The python files are there for debugging.

### Image Collection

I imagine a scenario where we'd want to collect images downrange. We didn't put that much effort obviously but could become important.

### Localization

Contains distance estimation and experimental implementations from papers that basically do info-constrained SLAM. Not a lot of immediately useful stuff in here other than distanceFromCamera.py.

### Mapping

Here is where we attempted publishing to ATAK. We settled for plotting on a Folium map last year. This will require a significant time investment.

Publishing and Mapping combine

