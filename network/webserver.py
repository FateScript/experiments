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


def parse_file_in_header(recv_header: str) -> str:
    return recv_header.split("\r\n")[0].split(" ")[1]


def mime_type(file: str) -> str:
    prefix = "Content-Type: "
    suffix = file.split(".")[-1]
    if suffix in ["html", "htm"]:
        return prefix + "text/html"
    elif suffix in ["pdf"]:
        return prefix + "application/pdf"
    elif suffix in ["jpeg", "jpg"]:
        return prefix + "image/jpeg"
    elif suffix in ["gif"]:
        return prefix + "image/gif"
    else:
        return prefix + "text/plain"


def text_404():
    return "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\n404 Not Found"


def text_500():
    return "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\n\r\n500 Internal Server Error"


def mime_and_data(file: str, encoding: str) -> str:
    prefix = os.path.realpath(".")
    full_path = os.path.sep.join([prefix, file])

    status = "HTTP/1.1 200 OK\r\n"
    if file == "/favicon.ico":  # favicon.ico to display
        fav_file = download_favicon()
        with open(fav_file, "rb") as f:
            icon_data = f.read()  # bytes data
        send_text = status + "Content-Type: image/x-icon\r\n\r\n" + icon_data.decode(encoding)
    elif file == "/":
        send_text = status + "Content-Type: text/plain\r\n\r\nHello world!"
    elif not os.path.exists(full_path):
        send_text = text_404()
    else:
        if not full_path.startswith(prefix):  # for security
            return text_404()

        try:
            mime_str = mime_type(file)
            with open(full_path, "rb") as f:
                data = f.read()
            encode_data = data.decode(encoding)
            length = len(data)
            send_text = status + mime_str + f"\r\nContent-Length: {length}\r\n\r\n" + encode_data
        except Exception:
            send_text = text_500()
    return send_text


def response_by_recv(recv: str, encoding: str) -> str:
    recv_header, *recv_body = recv.split("\r\n\r\n", maxsplit=1)
    print(f"Recv Header:\n{recv_header}\n")
    if recv_body:
        body_text = recv_body[0]
        if body_text:
            print(f"Recv Body:\n{body_text}\n")
    file = parse_file_in_header(recv_header)
    send_text = mime_and_data(file, encoding)
    return send_text


def server(port: int = 28333, encoding: str = "ISO-8859-1"):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen()
    print(f"Server is runing on http://localhost:{port}")
    while True:
        new_socket, addr_info = s.accept()
        data = new_socket.recv(4096)
        print(f"Addr info(ip, port): {addr_info}")
        decoded_data = data.decode(encoding)
        send_text = response_by_recv(decoded_data, encoding)

        new_socket.sendall(send_text.encode(encoding=encoding))
        new_socket.close()


if __name__ == "__main__":
    import fire
    fire.Fire(server)
