import threading
import cv2
from ultralytics import YOLO
import cvzone
import numpy as np
import math
import os
from time import strftime
import requests
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def first_model():
    video_path = "Video_fainting.mp4"
    telegram_token = "TOKEN FOR YOUR BOT"
    chat_id = "CHAT ID FOR YOUR BOT"
    image_path = "fall_detection.jpg"
    
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)
    times = {}

    def send_image(bot_token, chat_id, image):
        url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
        files = {'photo': open(image, 'rb')}
        data = {'chat_id': chat_id}
        response = requests.post(url, files=files, data=data)
        return response.json()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for consistency
        frame = cv2.resize(frame, (650, 400))

        # Predict the pose using YOLO model
        result = model.predict(frame)
        boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints = result[0].keypoints.data.cpu().numpy().astype("float")

        # Threshold to determine if an object is on the ground
        thresh = frame.shape[1] // 2 + 100

        for idd, lm in enumerate(keypoints):
            x1, y1, x2, y2 = boxes[idd]
            x2 = x2 - x1
            y2 = (y2 - y1)
            head = round(lm[5][0])
            legleft, legright = round(lm[16][0]), round(lm[15][0])

            # Detect if the person is in a fallen state
            if int(lm[16][0]) != 0 or int(lm[15][0]) != 0:
                if abs(head - legleft) > 60 and abs(head - legright) > 60 or x2 / y2 > 1:
                    if idd not in times:
                        times[idd] = 0
                    times[idd] += 1

                    # Draw a yellow rectangle around the detected object
                    cvzone.cornerRect(frame, (x1, y1, x2, y2), colorC=(0, 255, 255))

                    # Calculate the elapsed time
                    seconds = times[idd] // 1000
                    milliseconds = times[idd] % 1000
                    time_str = "{:02d}:{:02d}".format(seconds, milliseconds)
                    cv2.putText(frame, time_str, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

                    # If the object has been in a fallen state for more than 30 milliseconds, mark it
                    if int(milliseconds) > 30:
                        cv2.putText(frame, time_str, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                        cvzone.cornerRect(frame, (x1, y1, x2, y2), colorC=(0, 0, 255))

                        # Save the image of the detected object
                        image_to_save = frame[y1:y1 + y2, x1:x1 + x2]
                        cv2.imwrite(image_path, image_to_save)

                        # Send the image if the condition is met
                        if 32 > int(milliseconds) > 30:
                            send_image(telegram_token, chat_id, image_path)
                else:
                    # Reset the time if the object is not in a fallen state
                    cvzone.cornerRect(frame, (x1, y1, x2, y2))
                    times[idd] = 0

        # Display the frame
        cv2.imshow("Fall", frame)
        cv2.waitKey(1)

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def second_model():
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture("Video_cheeting.mp4")
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (650, 400))
        result = model.predict(frame)
        boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints = result[0].keypoints.data.cpu().numpy().astype("float")

        for idd, lm in enumerate(keypoints):
            x1, y1, x2, y2 = boxes[idd]
            x2 = x2 - x1
            y2 = (y2 - y1)
            distance = np.sqrt(int(lm[0][1])**2 + int(lm[16][1])**2)

            if int(lm[6][0]) + 20 > int(lm[0][0]) or int(lm[0][0]) > (int(lm[5][0]) - 20):
                cvzone.cornerRect(frame, (x1, y1, x2, y2), colorC=(0, 0, 255))
                count += 1
                cv2.putText(frame, f'{count}', (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 3)
            else:
                cvzone.cornerRect(frame, (x1, y1, x2, y2))

        cv2.imshow("Cheating", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


def third_model():
    # Load data from CSV files
    data1 = pd.read_csv(r"Fighting\violence.csv")
    data2 = pd.read_csv(r"Fighting\nonviolence.csv")
    
    # Create labels
    data1["51"] = 1
    data2["51"] = 0
    
    # Concatenate dataframes
    data3 = pd.concat([data1, data2], axis=0)
    data3 = data3.set_index(data3.columns[0])

    # Split data into features and labels
    x = data3.iloc[:, :-1]
    y = data3.iloc[:, -1]

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30, shuffle=True)

    # Train SVC model
    model = SVC()
    model.fit(x_train, y_train)

    # Save trained model
    with open(r"Fighting\\voilance-model.h5", "wb") as file:
        pickle.dump(model, file)

    # Load YOLO models
    detection_model = YOLO(r"yolov8n-pose.pt")
    svm_model = pickle.load(open(r"Fighting\\violence-model.h5", "rb"))

    # Open the video file
    cap = cv2.VideoCapture(r'Fighting\\istock2.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (650, 400))

        # Perform YOLO pose estimation on the frame
        result = detection_model(frame)
        boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)
        keypoints = result[0].keypoints.data.cpu().numpy().astype("float")

        # Iterate over the keypoints
        for lm in keypoints:
            if lm.shape[0] > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    k = []
                    for l in range(len(lm)):
                        k.append(lm[l][0])
                        k.append(lm[l][1])
                        k.append(lm[l][2])
                    reshaped_list = np.array([k])
                    # Check if there is a fight using the trained SVM model
                    if svm_model.predict(reshaped_list)[0] == 1:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "Fighting", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Display the frame
        cv2.imshow("Fighting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()


def fourth_model():
    # Load the YOLO model for fire detection
    model = YOLO("fire3_nano.pt")

    # Open the video file
    cap = cv2.VideoCapture("fire.mp4")

    # Labels for the detected objects
    names = ["Fire", "Smoke"]

    while True:
        # Read the frame
        _, image = cap.read()
        if image is None:
            break

        # Resize the frame
        image = cv2.resize(image, (650, 400))

        # Perform fire detection using the YOLO model
        result = model(image, stream=True)

        # Process the detection results
        for i in result:
            boxes = i.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Draw a rectangle around the detected object
                cvzone.cornerRect(image, (x1, y1, w, h))

                # Get the confidence score and class label
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                print(conf, names[cls])

                # Put text on the frame indicating the detected object
                cvzone.putTextRect(image, names[cls], (x1, y1), scale=2, colorB="blue")

        # Display the frame
        cv2.imshow("Fire_Detection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    thread1 = threading.Thread(target=first_model)
    thread2 = threading.Thread(target=second_model)
    thread3 = threading.Thread(target=third_model)
    thread4 = threading.Thread(target=fourth_model)
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    cv2.destroyAllWindows()
    
