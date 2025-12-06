#!/usr/bin/env python3

import os
import time
import subprocess


def stat_info(path):
    """Return file's dev + inode information"""
    try:
        st = os.stat(path)
        return f"dev={st.st_dev}, inode={st.st_ino}"
    except FileNotFoundError:
        return "(does not exist)"


def writer():
    """Continuously write content to file"""
    with open(file1, "w") as f:
        for i in range(10):
            f.write(f"line {i}\n")
            f.flush()
            print(f"[writer] wrote line {i}")
            time.sleep(0.4)
        print("[writer] finished writing")


def inode_check(file1, file2):
    # clean up, rm files if they exist
    for f in [file1, file2]:
        if os.path.exists(f):
            os.remove(f)

    pid = os.fork()

    if pid == 0:
        # Child process continuously writes
        writer()
        os._exit(0)
    else:
        time.sleep(1.2)  # Let child process write a few lines first
        print("\n=== inode info before move ===")
        print(f"{file1}", stat_info(file1))
        print(f"{file2}", stat_info(file2))

        print("\n>>> Executing file move !!!\n")
        subprocess.run(["mv", file1, file2])

        print("=== inode info after move ===")
        print(f"{file1}", stat_info(file1))  # Expected to not exist
        print(f"{file2}", stat_info(file2))  # inode should remain unchanged

        print("\nWaiting for writer to finish...\n")
        os.waitpid(pid, 0)

        print("\n=== Final file content ===")
        with open(file2) as f:
            print(f.read())


if __name__ == "__main__":
    file1 = "/tmp/test_a.txt"
    file2 = "/tmp/test_b.txt"
    inode_check(file1, file2)
