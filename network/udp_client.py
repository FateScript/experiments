#!/usr/bin/env python3

import socket


def udp_message(server: str, port: int, message: str):
    s = socket.socket(type=socket.SOCK_DGRAM)

    print("Sending message...")
    s.sendto(message.encode(), (server, port))

    data, sender = s.recvfrom(4096)
    server_ip, server_port = sender
    print(f"Got reply from {server_ip}:{server_port}: \"{data.decode()}\"")
    s.close()


if __name__ == "__main__":
    import fire
    fire.Fire(udp_message)
