# 🛠️ Real-Time Object Detection with YOLOv8 and OpenCV

This repository contains a project for **real-time object detection** using the **YOLOv8** model and **OpenCV**. The project demonstrates how to leverage a pre-trained YOLO model to detect various objects in a live video stream from a webcam. A special feature highlights **knives** with a **red bounding box** for easy identification.

## Project Overview

The primary goal of this project is to showcase the **real-time object detection** capabilities of the YOLOv8 model. YOLO (You Only Look Once) is renowned for its **speed** and **accuracy** in object detection tasks. This project includes the following steps:

1. **📹 Initializing the Webcam:** 
   - Setting up the webcam and configuring the resolution.

2. **🔍 Loading the YOLO Model:** 
   - Utilizing a pre-trained YOLOv8 model for object detection.

3. **📋 Defining Classes:** 
   - Listing the objects that the model can detect.

4. **🎥 Capturing and Processing Video Frames:** 
   - Reading frames from the webcam and processing them with the YOLO model to detect objects.

5. **✏️ Drawing Bounding Boxes:** 
   - Highlighting detected objects with bounding boxes and displaying their class names and confidence scores.

6. **🔪 Special Handling for Knives:** 
   - Drawing a **red bounding box** around knives to make them easily identifiable.

## Requirements

- **Python 3.x**
- **OpenCV**
- **Ultralytics YOLOv8**

Feel free to explore the repository and contribute to enhancing this real-time object detection system! 🚀

