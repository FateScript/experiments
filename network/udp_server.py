#!/usr/bin/env python3

import socket


def udp(port: int):
    s = socket.socket(type=socket.SOCK_DGRAM)
    s.bind(("", port))
    print(f"Listening on port {port}...")

    while True:
        data, sender = s.recvfrom(4096)
        ip, used_port = sender
        decode_text = data.decode()

        print(f"Got data from {ip}, port {used_port}: '{decode_text}'")
        s.sendto(f"Got your {len(data)} byte(s) of data!".encode(), sender)


if __name__ == "__main__":
    import fire
    fire.Fire(udp)
