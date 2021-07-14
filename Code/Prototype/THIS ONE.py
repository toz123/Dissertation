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
import threading

# --- Code to start the client

url = 'file:///D:/Users/Student/OneDrive/Bath/Dissertation/Code/Prototype/TestOrbit.html'

# initialise global variables

pause = True
speed = 0
widget_pos = [None, None]


def tracking():
    # --- Code to do the hand tracking
    # instantiate the methods for mp

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Drawing Hands and obtaining coordinates

    # create empty list to store all the pd dfs
    dataset, raw_list, processed_list = []

    cap = cv2.VideoCapture(0)
    start = time.time()

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=10) as hands:
        # initialise the frame count
        frame_counter = 0

        # don't understand why this is needed
        if not cap.isOpened():
            return

        while True:

            ret, frame = cap.read()

            if pause:
                continue

            # count the frame number manually
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
                # take out - save processing time
                for num, hand in enumerate(results.multi_hand_landmarks):

                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(
                                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(
                                                  color=(250, 44, 250), thickness=2, circle_radius=2),
                                              )

                # append widget frame data to list
                # add data to the list

            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    # dump all the widget data into a file here...

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

    # when the connection is closed save the dataset to a csv file

    with open('graphic.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(dataset)


# asynchronise function that speeds up and resets the widget
async def finish_test(websocket):

    global pause, speed

    # send message back to the server so that the speed of the widget is updated
    await asyncio.wait_for(websocket.send('speedup'), 1)

    pause = True

    # DOESNT WORK, need it so that the widget resets
    await asyncio.wait_for(websocket.send('reset'), 1)

# asynchronise function that calls finish_test 5 times


async def obtain(websocket, path):

    global pause, speed, widget_pos

    while True:

        data = await websocket.recv()

        if data == 'start':

            pause = False
            # initerate through 5 times

            if speed < 5:
                speed += 1

                # scheduling function in the future will do regardless of what is going on
                asyncio.get_event_loop().call_later(
                    20, lambda: asyncio.ensure_future(finish_test(websocket)))

            else:
                break

        # needed for data processing - global widget_pos becomes data being passed in
        elif "," in data:
            widget_pos = data


# Start the thread before calling the website and starting the server

thread1 = threading.Thread(target=tracking)
thread1.start()

# Code to start the websocket between client and server

webbrowser.open(url, new=0)
start_server = websockets.serve(obtain, host="localhost", port=9001)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
