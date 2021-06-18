# #!/usr/bin/env python
# # coding: utf-8

# set the python interpreter to 'D:\Skillshare\New folder\python.exe'

# import dependencies

import mediapipe as mp
import cv2
import numpy as np
import os
import pandas as pd
import time
import csv
import webbrowser
import websockets
import asyncio

# --- Code to start the client

# does not work in google chrome

url = 'file:///D:/Users/Student/OneDrive/Bath/Dissertation/Code/Prototype/TestOrbit.html'

webbrowser.open(url, new=0)

# ---Code to start the websocket between client and server

# delete the file if it doesn't exist
if os.path.exists('graphic.csv'):
    os.remove('graphic.csv')


async def obtain(websocket, path):
    # instantiate the dataset list
    dataset = []

    # add data to the list
    try:
        async for message in websocket:
            x_y_time = await websocket.recv()
            dataset.append(x_y_time)

    # when the connection is closed save the dataset to a csv file
    except websockets.exceptions.ConnectionClosed:

        with open('graphic.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(dataset)

start_server = websockets.serve(obtain, host="localhost", port=9001)


# --- Code to do the hand tracking

# need to edit so that the camera isn't visable

# instantiate the methods for mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Drawing Hands and obtaining coordinates

# create empty list to store all the pd dfs
raw_list = []
processed_list = []

cap = cv2.VideoCapture(0)
start = time.time()

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=10) as hands:
    # initialise the frame count
    frame_counter = 0
    while cap.isOpened():

        ret, frame = cap.read()

        # count the frame number manually as using cap.get(cv2.CAP_PROP_POS_FRAMES) is tidious
        frame_counter += 1

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        # checking if there are any hands in the frame
        if results.multi_hand_landmarks:

            # looping through each feature in the frame
            for num, hand in enumerate(results.multi_hand_landmarks):

                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(
                                              color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

            # Appending to the raw list - easier to use for the playback dataset
            raw_list.append(results.multi_hand_landmarks)

            # Appending to the processed list - easier to understand data

            # list comprehension tuple that contains feature, x, y, z coordinates for a particular frame
            df_frame = [[feature, coordinates.x, coordinates.y, coordinates.z]
                        for feature, coordinates in enumerate(results.multi_hand_landmarks[0].landmark)]

            # obtain whether the hand is left or right
            if results.multi_handedness[0].classification[0].label == 'Right':
                right_or_left = 1
            else:
                right_or_left = 0

            # measure the amount of time that has elapsed since the programme has started
            # this is to account for the fact that there might be time lags
            end = time.time()
            time_passed = end - start

            # generate a list of the frame count and what hand that is equivalent to the length of the df_frame
            frame_hand_time = [[frame_counter, time_passed,
                                right_or_left] for i in range(len(df_frame))]

            # reshape the array so that it's 1d
            convert_np = np.array(frame_hand_time)

            # convert to numpy array
            df_frame_np_array = np.array(df_frame)

            # concatenate the arrays together
            concate = np.concatenate((convert_np, df_frame_np_array), axis=1)

            # create a pd dataframe from the data
            df_appending = pd.DataFrame(
                concate, columns=['Frame', 'Time_Elapsed', 'Hand', 'Feature', 'x', 'y', 'z'])

            # append to a dataframe list - this will be merged later on
            processed_list.append(df_appending)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# Export Datasets

path = r'D:\Users\Student\OneDrive\Bath\Dissertation\Data\MediaPipe_Implementation'


def exporter(dataset, location, export_name):
    '''
    Function that exports the dataset
    Args.
    Dataset - the dataset to export
    Location - the location to export the dataset to
    Export_name - the name of the file saved
    '''

    path = location
    dataset.to_csv(os.path.join(path, export_name))


# Raw data export

# opening the csv file in 'w+' mode
file = open(path + '\\' + 'raw_data.csv', 'w+', newline='')

# writing the data into the file
with file:
    write = csv.writer(file)
    write.writerows(raw_list)

# Processed dataset

# merge the dataframe into one set if results exist
if len(processed_list) > 0:
    merged_df = pd.concat(processed_list)

# set the index as the frame
merged_df.set_index('Frame')

exporter(merged_df, path, 'process_data.csv')

# Playback - save as an video

path_videos = r'D:\Users\Student\OneDrive\Bath\Dissertation\Data\Replay_videos'

# create blank opencv2 matrix
blank_img = np.zeros([512, 512, 3], dtype=np.uint8)
blank_img.fill(0)
height, width, layers = blank_img.shape
img_array = []

# loop through all the results in the raw list
for i in range(len(raw_list)):

    # reinstantiate the blank image for each iteration
    annotated_image = blank_img.copy()

    # loop through each hand of i'th element
    for hand in raw_list[i]:

        # draw the landmarks on the image
        # takes the image
        # the hand - list of landmarks
        # the connections between the features - mp_hands.HAND_CONNECTIONS
        mp_drawing.draw_landmarks(annotated_image,
                                  hand, mp_hands.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(250, 44, 250), thickness=2, circle_radius=2),
                                  )

    img_array.append(annotated_image)

out = cv2.VideoWriter(os.path.join(
    path_videos, 'video1.avi'), 0, 30, (width, height))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
