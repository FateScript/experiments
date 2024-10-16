#!/usr/bin/env python3

import argparse
import json
import select
import socket
import threading
from typing import Dict

from chatui import end_windows, init_windows, print_message, read_command


class Client:

    def __init__(self, name, host: str = "localhost", port: int = 12211):
        self.name = name
        self.host = host
        self.port = port

        self.connect()

    def connect(self, send_hello: bool = True):
        sock = socket.socket()
        sock.connect((self.host, self.port))
        self.sock = sock
        self.sock_list = [self.sock]
        if send_hello:
            self.send_hello()
        return self.sock

    def send_message(self, data_dict):
        send_text = json.dumps(data_dict).encode("utf-8")
        self.sock.send(send_text)

    def send_hello(self):
        hello_message = {
            "type": "hello",
            "nick": self.name,
        }
        self.send_message(hello_message)

    def send_chat(self, message):
        chat_message = {
            "type": "chat",
            "message": message,
        }
        self.send_message(chat_message)

    def send_join(self):
        join_message = {
            "type": "join",
            "nick": self.name,
        }
        self.send_message(join_message)

    def send_leave(self):
        leave_message = {
            "type": "leave",
            "nick": self.name,
        }
        self.send_message(leave_message)

    def split_message(self, message: str):
        # to solve tcp packet reassembly
        try:
            data_dict = json.loads(message)
            self.handle_messages(data_dict)
        except json.JSONDecodeError:  # maybe multiple messages were sent together
            if message.count("}{") > 0:
                messages = message.split("}{")
                for idx, msg in enumerate(messages):
                    if idx == 0:
                        msg += "}"
                        self.handle_messages(json.loads(msg))
                    elif idx == len(messages) - 1:
                        msg = "{" + msg
                        self.handle_messages(json.loads(msg))
                    else:
                        msg = "{" + msg + "}"
                        self.handle_messages(json.loads(msg))

    def handle_messages(self, message: Dict):
        name = message["nick"]
        if message["type"] == "join":
            if name == self.name:
                # self.sock_list.append(self.sock)
                print_message("You have joined")
            else:
                print_message(f"{name} join")
        elif message["type"] == "leave":
            if name == self.name:
                print_message("You have left")
            else:
                print_message(f"{name} has left")
        elif message["type"] == "chat":
            print_message(f"{name}: {message['message']}")

    def runner(self):
        while True:
            quit_flag = False

            read_sockets, _, _ = select.select(self.sock_list, [], [])
            for s in read_sockets:
                recv_text = s.recv(1024)
                json_text = recv_text.decode("utf-8")
                if not json_text:  # empty means connection is closed, terminate the thread
                    quit_flag = True
                    self.sock_list.remove(s)
                    break

                self.split_message(json_text)

            if quit_flag:
                break

    def background_thread(self):
        t = threading.Thread(target=self.runner, daemon=True)
        return t

    def run(self):
        thread = self.background_thread()
        thread.start()

        while True:
            chat_message = read_command("Enter >>> ").strip()
            if chat_message == "/leave":
                self.send_leave()
                thread.join()
            elif chat_message == "/join":
                self.connect(send_hello=False)
                thread = self.background_thread()
                thread.start()
                self.send_hello()
                self.send_join()
            else:  # normal chat message
                if self.sock_list:  # there is a valid connection
                    self.send_chat(chat_message)


def make_parser():
    parser = argparse.ArgumentParser("Chat client")
    parser.add_argument("name", help="Your nickname")
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=12211, help="Server port")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    host, port = args.host, args.port

    init_windows()

    client = Client(args.name, host=args.host, port=args.port)
    client.run()

    end_windows()
