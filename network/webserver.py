#!/usr/bin/env python3

import socket


def server(port: int = 28333):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen()
    print(f"Server is runing on http://localhost:{port}")
    while True:
        new_socket, addr_info = s.accept()
        print(f"Addr info: {addr_info}")
        data = new_socket.recv(4096)
        print(f"Recv data:\n{data.decode('utf-8')}")
        send_text = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<h1>Hello world!</h1>"
        new_socket.sendall(send_text.encode("ISO-8859-1"))
        new_socket.close()


if __name__ == "__main__":
    import fire
    fire.Fire(server)
