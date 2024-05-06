#!/usr/bin/env python3

import os
import socket
import webbrowser


def connect(url: str, port: int = 80, encoding: str = "ISO-8859-1") -> str:
    # NOTE: To connect localhost, use 'localhost' as the url.
    s = socket.socket()
    s.connect((url, port))
    # If you were requesting a specific file, it would be on that first line, for example:
    # GET /path/to/file.html HTTP/1.1
    send_text = f"GET / HTTP/1.1\r\nHost: {url}\r\nConnection: close\r\n\r\n"
    # if the data is not encoded with ISO-8859-1,
    # you’ll get weird characters in your string or an error.
    # For example.com, send_text.encode() also works.
    s.sendall(send_text.encode(encoding=encoding))

    block_size = 4096
    text = ""
    while True:
        data = s.recv(block_size)
        text += data.decode("utf-8")
        if len(data) < block_size:
            break
    s.close()
    return text


def connect_and_display(url: str = "example.com", port: int = 80, display: bool = True):
    text = connect(url, port)
    header, web_text = text.rsplit("\r\n", maxsplit=1)
    print(f"Header:\n{header}\n")
    print(f"Body:\n{web_text}")
    with open(f"{url}.html", "w") as f:
        f.write(web_text)

    if display:
        webbrowser.open("file://" + os.path.realpath(f"{url}.html"))


if __name__ == "__main__":
    import fire
    fire.Fire(connect_and_display)
