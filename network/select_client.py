# Example usage:
#
# python3 select_client.py alice localhost 3490
# python3 select_client.py bob localhost 3490
# python3 select_client.py chris localhost 3490
#
# The first argument is a prefix that the server will print to make it
# easier to tell the different clients apart. You can put anything
# there.

import sys
import socket
import time
import random


def usage():
    print("usage: select_client.py prefix host port", file=sys.stderr)


def random_string():
    """ Returns a random string of ASCII printable characters. """

    length = random.randrange(10, 20)
    s = ""

    for _ in range(length):
        codepoint = random.randint(97, 122)
        s += chr(codepoint)

    return s


def delay_random_time():
    delay_seconds = random.uniform(1, 5)
    time.sleep(delay_seconds)


def main(argv):
    try:
        prefix = argv[1]
        host = argv[2]
        port = int(argv[3])
    except Exception:
        usage()
        return 1

    # Make the client socket and connect
    s = socket.socket()
    s.connect((host, port))

    # Loop forever sending data at random time intervals
    while True:
        string_to_send = f"{prefix}: {random_string()}"
        string_bytes = string_to_send.encode()
        s.send(string_bytes)

        delay_random_time()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
