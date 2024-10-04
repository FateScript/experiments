# Example usage:
#
# python3 select_server.py 3490

import sys
import socket
import select


def run_server(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen()

    connected_set = []
    connected_set.append(s)

    while True:
        ready_set, _, except_set = select.select(connected_set, {}, connected_set)
        for sock in ready_set:
            # check if sock is listener socket
            if sock is s:
                new_socket, addr_info = s.accept()
                print(f"{addr_info}: connected")
                connected_set.append(new_socket)
            else:  # sock is a regular socket
                data = sock.recv(1024)
                byte_length = len(data)
                peer_name = sock.getpeername()
                if byte_length == 0:
                    print(f"{peer_name}: disconnected")
                    sock.close()
                    connected_set.remove(sock)
                    continue
                print(f"Received {byte_length} bytes from {peer_name}: {data.decode()}")

        for sock in except_set:
            print(f"Exceptional condition on {sock.getpeername()}")
            sock.close()
            connected_set.remove(sock)


def usage():
    print("usage: select_server.py port", file=sys.stderr)


def main(argv):
    try:
        port = int(argv[1])
    except Exception:
        usage()
        return 1

    run_server(port)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
