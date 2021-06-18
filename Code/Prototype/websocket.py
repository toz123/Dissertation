import websockets
import asyncio
import csv
import os
import webbrowser

# delete the file if it doesn't exist
if os.path.exists('graphic.csv'):
    os.remove('graphic.csv')

# does not work in google chrome

url = 'file:///D:/Users/Student/OneDrive/Bath/Dissertation/Code/Prototype/TestOrbit.html'

webbrowser.open(url, new=0)


async def obtain(websocket, path):
    # instantiate the dataset list
    dataset = []
    #
    # send request for coordinates
    # websocket.send --> display coordinates

    # add data to the list

    # want to integrate with handtracking - should be on every frame that is executing
    try:
        async for message in websocket:
            x_y_time = await websocket.recv()
            dataset.append(x_y_time)

    # should be done when
    # when the connection is closed save the dataset to a csv file
    except websockets.exceptions.ConnectionClosed:

        with open('graphic.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(dataset)

start_server = websockets.serve(obtain, host="localhost", port=9001)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
