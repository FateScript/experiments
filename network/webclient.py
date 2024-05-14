#!/usr/bin/env python3

import os
import socket
import webbrowser
from typing import Tuple


def parse_url(url: str) -> Tuple[str, int, str]:
    """Parse the URL to get the host, port, and path."""
    http_type = "http://" in url
    prefix = "http://" if http_type else "https://"
    url = url.replace(prefix, "")
    if url.startswith("localhost"):
        prefix = ""

    if ":" in url:
        url, port_path = url.split(":")
        if "/" in port_path:
            port, path = port_path.split("/", maxsplit=1)
        else:
            port, path = port_path, ""
        return prefix + url, int(port), "/" + path
    else:
        return prefix + url, 80, "/"


def connect(url: str, encoding: str = "ISO-8859-1") -> str:
    # NOTE: To connect localhost, use 'localhost' as the url.
    url, port, path = parse_url(url)
    print(f"URL: {url}\tPort: {port}\nPath: {path}")
    s = socket.socket()
    s.connect((url, port))
    # If you were requesting a specific file, it would be on that first line, for example:
    # GET /path/to/file.html HTTP/1.1
    send_text = f"GET {path} HTTP/1.1\r\nHost: {url}\r\nConnection: close\r\n\r\n"
    # if the data is not encoded with ISO-8859-1,
    # you’ll get weird characters in your string or an error.
    # For example.com, send_text.encode() also works.
    s.sendall(send_text.encode(encoding=encoding))

    block_size = 4096
    text = bytearray()
    while True:
        data = s.recv(block_size)
        text.extend(data)
        if len(data) < block_size:
            break
    s.close()
    text = text.decode("utf-8")
    return text


def connect_and_display(url: str = "example.com", display: bool = False):
    text = connect(url)
    header, web_text = text.rsplit("\r\n", maxsplit=1)
    print(f"Header:\n{header}\n")
    print(f"Body:\n{web_text}")

    *_, file = parse_url(url)
    file = os.path.basename(file)
    save_file = file if file else "file.html"
    with open(save_file, "w") as f:
        f.write(web_text)

    if display:
        webbrowser.open("file://" + os.path.realpath(save_file))


if __name__ == "__main__":
    import fire
    fire.Fire(connect_and_display)
