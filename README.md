Drowsiness Detection

## Description
The project uses the Flask library to deloy and uses the YOLOv8 machine learning algorithm for object detection, specifically targeting the identification of human faces. The purpose of the project is to deal with driving incidents when the driver fell asleep.

## Video demo


https://github.com/user-attachments/assets/a4d1f747-009b-4d30-b164-ffa492147819



## Features
### Object Detection: The application utilizes the trained YOLOv8 model to detect drowsiness face in real-time using the webcam feed.

### Custom Dataset: The model has been trained on a custom dataset specific to drowsiness face.
![image](https://github.com/user-attachments/assets/f221fd89-4c2d-475a-a3c1-830467be2827)

### Flask Web Application: The project is implemented as a Flask web application.
![image](https://github.com/user-attachments/assets/0f989486-78f1-4d43-b45b-b2347bcde5c5)

## Installation
1. Clone the project repository:
 `git clone`
 
2. Install the required dependencies:
 `pip install -r requirements.txt`
 
## Usage
1. Start the Flask application:
 `python flaskapp.py`
 
2. Access the application through the provided URL (e.g., http://localhost:5000).

3. Grant necessary permissions to access the webcam.

4. Once the webcam feed is displayed, the application will automatically detect and highlight drowsiness face in the video stream.

5. You can also upload video and image to detect instead
