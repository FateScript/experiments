
## Network

### Files

`webclient.py`: The main file that contains the implementation of the web client.  
`webserver.py`: The main file that contains the implementation of the web server.  

`udp_server.py`: implementation of the UDP server.
`udp_client.py`: implementation of the UDP client.

### Usage

#### Start web server
```
python3 webserver.py -p #port-number
```

#### Start web client
```
python3 webclient.py --url http://your-server-ip:port-number
```

#### Trace packet

Run the following command to trace the packet on the loopback address, port 12345.
```
sudo tcpdump -i lo0 port 12345 -v
```
