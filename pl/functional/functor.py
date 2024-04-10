#!/usr/bin/env python3

import itertools
from dataclasses import dataclass
from typing import Any, Callable

from pytools.functional import compose, curry

__all__ = [
    "Functor", "Maybe", "Either", "Left", "Right",
]


@dataclass
class Functor:
    """A functor is an object that can be mapped over."""
    value: Any

    def map(self, func):
        return self.of(func(self.value))

    @classmethod
    def of(cls, value):
        return cls(value=value)

    def ap(self, functor):
        return functor.map(self.value)

    def join(self):
        return self.value

    def chain(self, func):
        return self.map(func).join()


class Maybe(Functor):
    """A Maybe is a functor that might have no value."""

    def map(self, func):
        if self.is_nothing:
            return self
        return self.of(func(self.value))

    def join(self):
        return self if self.is_nothing else self.value

    @property
    def is_nothing(self):
        return self.value is None


class Either(Functor):

    @classmethod
    def of(cls, value):
        return Right.of(value)


class Left(Either):
    """A Left is a functor that holds the left value."""

    def map(self, func):
        return self

    @classmethod
    def of(cls, value):
        return cls(value=value)


class Right(Either):
    """A Right is a functor that holds the right value."""

    def map(self, func):
        return Either.of(func(self.value))

    @classmethod
    def of(cls, value):
        return cls(value=value)


@curry
def fmap(func, functor):
    return functor.map(func)


@curry
def either(left: Callable, right: Callable, functor: Any):
    if isinstance(functor, Left):
        return functor.map(left)
    elif isinstance(functor, Right):
        return functor.map(right)


def left(value) -> Left:
    return Left.of(value)


@curry
def maybe(v, f, maybe: Maybe):
    if maybe.is_nothing:
        return v
    else:
        return maybe.map(f)


def join(functor):
    return functor.join()


@curry
def chain(func, functor):
    return functor.chain(func)


def identity(x):
    return x


def test_associativity_law():
    # associativity law: compose(join, map(join)) == compose(join, join);
    values = [3, list(range(10)), "Hello world"]
    As = [Functor, Maybe, Either]
    for val in values:
        for A, B, C in itertools.product(As, As, As):
            v = A.of(B.of(C.of(val)))
            assert compose(join, fmap(join))(v) == compose(join, join)(v)

    print("pass associativity law!")


def test_identity_for_functor():
    # // identity for all functor like (M a): compose(join, of) == compose(join, map(of)) == id;
    values = [3, list(range(10)), "Hello world"]
    As = [Functor, Maybe, Either]
    for A, val in zip(As, values):
        v = A.of(val)
        assert compose(join, A.of)(v) == compose(join, fmap(A.of))(v)
        assert compose(join, A.of)(v) == identity(v)

    print("pass identity for Functor!")


def test_identity_law():
    # identity law: A.of(identity).ap(v) == v
    values = [
        Functor.of(3),
        Maybe.of(4),
        Either.of(5),
    ]
    for v in values:
        A = type(v)
        assert A.of(identity).ap(v) == v

    print("pass identity law!")


def test_homomorphism_law():
    # homomorphism law: A.of(f).ap(A.of(x)) == A.of(f(x))
    fs = [
        lambda x: x + 1,
        lambda x: x[:3],
        lambda x: x.split(" "),
    ]
    values = [3, list(range(10)), "Hello world"]
    As = [Functor, Maybe, Either]
    for f, v, A in zip(fs, values, As):
        assert A.of(f).ap(A.of(v)) == A.of(f(v))

    print("pass homomorphism law!")


def test_interchange_law():
    # interchange law: v.ap(A.of(x)) == A.of(f => f(x)).ap(v)
    fs = [
        Functor.of(lambda x: x + 1),
        Maybe.of(lambda x: x[:3]),
        Either.of(lambda x: x.split(" ")),
    ]
    values = [3, list(range(10)), "Hello world"]
    for v, x in zip(fs, values):
        A = type(v)
        assert v.ap(A.of(x)) == A.of(lambda f: f(x)).ap(v)

    print("pass interchange law!")


def test_composition_law():
    # composition law: A.of(compose).ap(u).ap(v).ap(w) == u.ap(v.ap(w))

    @curry
    def add(x, y):
        return x + y

    for A in [Functor, Maybe, Either]:
        u, v, w = A.of(add), A.of(3), A.of(4)
        # NOTE: different from u.ap(v.ap(w)) here due to implemntion of `compose`
        assert A.of(compose).ap(u).ap(v).ap(w) == u.ap(v).ap(w)

    for A in [Functor, Maybe, Either]:
        f, g = add(2), add(3)
        u, v, w = A.of(f), A.of(g), A.of(4)
        assert A.of(compose(f, g)).ap(w) == u.ap(v.ap(w))

    print("pass composition law!")


if __name__ == "__main__":
    test_associativity_law()
    test_identity_for_functor()
    test_identity_law()
    test_homomorphism_law()
    test_interchange_law()
    test_composition_law()
