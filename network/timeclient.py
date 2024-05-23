#!/usr/bin/env python3

import socket
import time


def nist_time() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    time_since_1900 = 0

    while time_since_1900 == 0:
        host = "time.nist.gov"
        port = 37
        s.connect((host, port))
        data = s.recv(4)  # 4 bytes represent time since 1900.1.1, 0 represents failed
        s.close()
        time_since_1900 = int.from_bytes(data, byteorder="big")

    return time_since_1900


# seconds between 1900-01-01 and 1970-01-01
second_delta = 2208988800

time_since_1900 = nist_time()
system_time = int(time.time()) + second_delta

print(f"NIST time  : {time_since_1900}")
print(f"System time: {system_time}")
print(f"Delta: {time_since_1900 - system_time}")
