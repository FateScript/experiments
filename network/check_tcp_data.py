#!/usr/bin/env python3

# download the tcp data from https://beej.us/guide/bgnet0/source/exercises/tcpcksum/tcp_data.zip
# unzip the file and and run this script in the same directory.


def read_data(index: int = 0):
    with open(f"tcp_addrs_{index}.txt", "r") as f:
        address = f.read().split()
    with open(f"tcp_data_{index}.dat", "rb") as f:
        content = f.read()
    return address, content


def ip_to_bytes(ip: str) -> bytes:
    byte_data = [int(x).to_bytes(1, byteorder="big") for x in ip.split(".")]
    return bytes(b"").join(byte_data)


def tcp_pseudo_header(from_ip: str, to_ip: str, tcp_length: int) -> bytes:
    # tcp header format
    # +--------+--------+--------+--------+
    # |           Source Address          |
    # +--------+--------+--------+--------+
    # |         Destination Address       |
    # +--------+--------+--------+--------+
    # |  Zero  |  PTCL  |    TCP Length   |
    # +--------+--------+--------+--------+
    from_bytes, to_bytes = ip_to_bytes(from_ip), ip_to_bytes(to_ip)
    zero, protocol = b"\x00", b"\x06"
    length_bytes = tcp_length.to_bytes(2, byteorder="big")
    return from_bytes + to_bytes + zero + protocol + length_bytes


def tcp_checksum(tcp_data: bytes) -> bytes:
    offset, ck_sum = 0, 0

    while offset < len(tcp_data):
        word = tcp_data[offset:offset + 2]
        word_value = int.from_bytes(word, byteorder="big")
        ck_sum += word_value
        ck_sum = (ck_sum & 0xffff) + (ck_sum >> 16)
        offset += 2

    ck_sum = (~ck_sum) & 0xffff
    return ck_sum.to_bytes(2, byteorder="big")


def check():
    # tcp diagram, port number is 2 bytes, sequence/ack number is 4 bytes
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |          Source Port          |       Destination Port        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                        Sequence Number                        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                    Acknowledgment Number                      |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |  Data |           |U|A|P|R|S|F|                               |
    # | Offset| Reserved  |R|C|S|S|Y|I|            Window             |
    # |       |           |G|K|H|T|N|N|                               |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |           Checksum            |         Urgent Pointer        |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                    Options                    |    Padding    |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # |                             data                              |
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    for idx in range(10):
        (from_ip, to_ip), tcp_data = read_data(idx)
        cksum_origin = tcp_data[16:18]

        tcp_length = len(tcp_data)

        pesudo_header = tcp_pseudo_header(from_ip, to_ip, tcp_length)
        zero_cksum_tcp = tcp_data[:16] + b"\x00\x00" + tcp_data[18:]
        if tcp_length % 2 == 1:  # pad zero to make the length even
            zero_cksum_tcp += b"\x00"
        ck_sum = tcp_checksum(pesudo_header + zero_cksum_tcp)
        if ck_sum == cksum_origin:
            print("PASS")
        else:
            print("FAIL")


if __name__ == "__main__":
    check()
