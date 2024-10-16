#!/usr/bin/env python3

import argparse
import json
import select
import socket
from typing import Dict, Tuple


class Server:

    def __init__(self, port: int = 12211):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', port))
        sock.listen()
        print(f"Server running at port {port}")

        self.port = port
        self.sock = sock
        self.client_sock_map: Dict[socket.socket, Tuple[str, int]] = {}
        self.sock_name_map: Dict[socket.socket, str] = {}
        self.connected_set = [self.sock]

    def add_client(self, socket):
        peer_name = socket.getpeername()
        if socket in self.client_sock_map:
            print(f"Client {peer_name} already exists, overwriting")
        self.client_sock_map[socket] = peer_name
        self.connected_set.append(socket)

    def rm_client(self, socket):
        peer_name = self.client_sock_map.pop(socket, None)
        self.sock_name_map.pop(socket, None)
        self.connected_set.remove(socket)
        socket.close()
        print(f"Client {peer_name} removed")

    def notify_all_clients(self, data: Dict, exclude_sock: socket.socket = None):
        send_text = json.dumps(data)
        send_bytes = send_text.encode("utf-8")
        for sock, name in self.sock_name_map.items():  # notify all clients that said "hello"
            if sock == exclude_sock:
                continue
            print(f"Send {send_text} to {name}{sock.getpeername()}")
            sock.send(send_bytes)

    def dispatch_message(self, message: str, sock: socket.socket):
        # to solve tcp packet reassembly
        try:
            data_dict = json.loads(message)
            self.handle_messages(data_dict, sock)
        except json.JSONDecodeError:  # maybe multiple messages were sent together
            if message.count("}{") > 0:
                messages = message.split("}{")
                for idx, msg in enumerate(messages):
                    if idx == 0:
                        msg += "}"
                        self.handle_messages(json.loads(msg), sock)
                    elif idx == len(messages) - 1:
                        msg = "{" + msg
                        self.handle_messages(json.loads(msg), sock)
                    else:
                        msg = "{" + msg + "}"
                        self.handle_messages(json.loads(msg), sock)

    def handle_messages(self, data_dict, sock):
        data_type = data_dict["type"]
        if data_type == "hello":
            self.handle_hello(data_dict, sock)
        elif data_type == "chat":
            self.handle_chat(data_dict, sock)
        elif data_type == "join":
            self.handle_join(data_dict, sock)
        elif data_type == "leave":
            self.handle_leave(data_dict, sock)
        else:
            raise ValueError(f"Unknown message type: {data_type}")

    def handle_hello(self, data_dict, sock):
        name = data_dict["nick"]
        if sock in self.sock_name_map:
            print(f"Client {name} already exists, overwriting")
        self.sock_name_map[sock] = name

    def handle_chat(self, data_dict, sock):
        name = self.sock_name_map[sock]
        data_dict["nick"] = name
        self.notify_all_clients(data_dict)

    def handle_join(self, data_dict, sock):
        name = data_dict["nick"]
        self.add_client(sock)
        self.sock_name_map[sock] = name
        self.notify_all_clients(data_dict)

    def handle_leave(self, data_dict, sock):
        self.notify_all_clients(data_dict)
        self.rm_client(sock)

    def run_server(self):
        while True:
            ready_set, _, except_set = select.select(self.connected_set, {}, self.connected_set)
            for sock in ready_set:
                if sock is self.sock:  # check if sock is listener socket
                    new_socket, addr_info = self.sock.accept()
                    print(f"{addr_info}: connected")
                    self.add_client(new_socket)
                else:  # sock is a regular socket
                    name = self.sock_name_map.get(sock, "Unknown")
                    try:
                        data = sock.recv(1024, socket.MSG_DONTWAIT)
                    except BlockingIOError:  # hangup
                        continue
                    except OSError:
                        print(f"{name}: disconnected")
                        self.rm_client(sock)
                        continue

                    peer_name = sock.getpeername()
                    byte_length = len(data)
                    if byte_length == 0:
                        print(f"{name}{peer_name}: disconnected")
                        self.rm_client(sock)
                        continue

                    message = data.decode("utf-8")
                    print(f"Received {byte_length} bytes from {name}{peer_name}: {message}")
                    self.dispatch_message(message, sock)

            for sock in except_set:
                print(f"Exceptional condition on {sock.getpeername()}")
                sock.close()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12211)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    s = Server(port=args.port)
    s.run_server()
