
## Chat Server and Client

A pure python chat server and client implementation, see [link](https://beej.us/guide/bgnet0/html/split/project-multiuser-chat-client-and-server.html#project-multiuser-chat-client-and-server) to get more details.

### Usage

#### Start chat server

run the following command to start the chat server.
```
python3 chat_server.py
```

#### Start chat client
run the following command to start multiple clients, alice/bob is the chat nick name.
```
python3 chat_client.py alice
python3 chat_client.py bob
```

#### Chat commands

`/leave`: leave the chat room, chat messages will not be received any more.  
`/join`: join the chat room, chat messages will be received.  
press ctrl+c to exit the chat client (The server will log disconnected).
