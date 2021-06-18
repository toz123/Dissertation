// kill the server - taskkill /F /IM node.exe

// needed to save the data to a json file
const fs = require('fs');

// this is the code for the server - using websockets
// https://www.youtube.com/watch?v=FduLSXEHLng - guide used to help create the server

const WebSocket = require('ws');
const wss = new WebSocket.Server({port: 9001});

// log when there is a connection and when the connection is closed
wss.on('connection', ws => {

    console.log("New client connected!");

    // clear the array every time a new client connects to the server
    var arr = [];
    
    // message sent from client to server
    ws.on("message", data => {
        arr.push(data)
        // console.log('Client has sent data: ' + data);
    });

    ws.on('close', () => {
        // stringify JSON Object
        jsonArr = JSON.stringify(arr)        
        // console.log(jsonArr);
        // write to JSON file
        fs.writeFile('file.txt', jsonArr, (err) => {
            if(err) {
                throw err;
            }
            console.log("Data has been written to file successfully.");
        });
    });
});
