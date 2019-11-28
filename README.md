# video_place_recognition
VTT 3rd year

# place recognition software for Friends video based on scene change detector
This repository contains codes and pre-trained checkpoints of a place recognition model and scene change detector for Friend video.

# Requirements
 - Ubuntu 16.04
 - Python 2.7
 - Pytorch 1.2.0
 - NumPy
 - PIL
 - opencv-python
 - matplotlib
 - jsonl

# Download pre-trained checkpoint
Pre-trained checkpoint should be placed in the root directory.
You can download the checkpoint at here " "

# How to use
Input : video file (e.g. *.avi, *.mkv)
Output : jsonl file (predicted class of video frames for every second, 1fps)

Below is an example of output jsonl file.

    {"second": 0.0, "type": "location", "class": "none"}
    {"second": 0.0, "type": "location", "class": "none"}
    ...
    {"second": 52.0, "type": "location", "class": "cafe"}
    {"second": 53.0, "type": "location", "class": "cafe"}
    ...
    {"second": 314.0, "type": "location", "class": "home-livingroom-Monica"}
    {"second": 315.0, "type": "location", "class": "home-livingroom-Monica"}

You can run following command on a terminal,

    python demo.py --input_filename <video-file> --output_filename <output-file-name>
    
For example 

    python demo.py --input_filename input.mkv --output_filename output
    
Then output.jsonl file will be saved in the root directory

# Acknowledgements
This project was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)

