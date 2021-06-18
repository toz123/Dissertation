import websockets
import asyncio
import csv
import os

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

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
