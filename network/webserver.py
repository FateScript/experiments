#!/usr/bin/env python3

import os
import socket


def download_favicon(save_name: str = "favicon.ico") -> str:
    import requests
    if os.path.exists(save_name):
        return save_name

    url = "https://raw.githubusercontent.com/FateScript/FateScript.github.io/master/favicon.ico"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_name, 'wb') as f:
            f.write(response.content)
            print("Download successfully!")
    else:
        print("Failed to download!")
    return save_name


def is_favicon(recv_header: str) -> bool:
    return recv_header.split("\r\n")[0].split(" ")[1] == "/favicon.ico"


def server(port: int = 28333, encoding: str = "ISO-8859-1"):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen()
    print(f"Server is runing on http://localhost:{port}")
    while True:
        new_socket, addr_info = s.accept()
        print(f"Addr info(ip, port): {addr_info}")
        data = new_socket.recv(4096)
        decoded_data = data.decode(encoding)
        recv_header, *recv_body = decoded_data.split("\r\n\r\n", maxsplit=1)
        print(f"Recv Header:\n{recv_header}\n")
        if recv_body:
            body_text = recv_body[0]
            if body_text:
                print(f"Recv Body:\n{body_text}\n")
        # send_text = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>Hello world!</h1>"
        if is_favicon(recv_header):
            fav_file = download_favicon()
            with open(fav_file, "rb") as f:
                icon_data = f.read()
            send_text = "HTTP/1.1 200 OK\r\nContent-Type: image/x-icon\r\n\r\n" + icon_data.decode(encoding)
        else:
            send_text = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHello world!"

        new_socket.sendall(send_text.encode(encoding=encoding))
        new_socket.close()


if __name__ == "__main__":
    import fire
    fire.Fire(server)
