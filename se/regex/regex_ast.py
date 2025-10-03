#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

__all__ = [
    "Node",
    "State",
    "tokenize",
    "ast",
    "node_to_state",
    "regex_to_states",
    "replace_success_state",
    "reset_state",
]


TOKEN_TYPES = {
    "STAR": "*",
    "OPTIONAL": "?",
    "OR": "|",
    "PLUS": "+",
    "LPAREN": "(",
    "RPAREN": ")",
    "LITERAL": str.isalpha,
}


def tokenize(regex: str):
    i = 0
    while i < len(regex):
        char = regex[i]
        for token_type, token_val in TOKEN_TYPES.items():
            if isinstance(token_val, str):
                if char == token_val:
                    yield token_type, char
            elif callable(token_val):
                if token_val(char):
                    yield token_type, char
            else:
                yield "UNKNOWN", char
        i += 1


def right_most_node(node: Node) -> Node:
    # set node2 to the right most of node1
    cur_node = node
    while cur_node.right is not None:
        cur_node = cur_node.right
    return cur_node


@dataclass
class Node:

    symbol: str = ""
    left: Node | None = None
    right: Node | None = None
    symbol_type: str = "UNKNOWN"

    def concat_node(self, node: Node):
        cur_node = right_most_node(self)
        cur_node.right = node
        return self


@dataclass
class RepeatNode(Node):

    def concat_node(self, node: Node):
        self.left = node
        return self


def is_lparen_node(node: Node) -> bool:
    return node.symbol_type == "LPAREN"


def merge_stack_to_one_node(stack: list[Node]) -> Node:
    # deque should be better
    while len(stack) > 1:
        node1 = stack.pop(0)
        node2 = stack.pop(0)
        right_most = right_most_node(node1)
        right_most.right = node2
        # node1.concat_node(node2)
        stack.insert(0, node1)
    return stack[0]


def merge_node_rparen_type(stack) -> Node:
    idx = len(stack) - 1
    nodes = []
    while stack[idx].symbol_type != "LPAREN":
        node = stack.pop()
        nodes.insert(0, node)
        if idx == 0:
            raise ValueError("Invalid syntax, Mismatched parentheses")
        idx -= 1
    pop_node = stack.pop()  # pop the '('
    assert pop_node.symbol_type == "LPAREN"
    node = merge_stack_to_one_node(nodes)
    return node


def merge_node_or_type(stack) -> Node:
    idx = len(stack) - 1
    nodes = []
    while idx >= 0 and stack[idx].symbol_type != "LPAREN":
        node = stack.pop()
        nodes.insert(0, node)
        idx -= 1
    node = merge_stack_to_one_node(nodes)
    return node


def ast(tokens: list[tuple[str, str]]) -> Node:
    # Abstract Syntax Tree construction for regex
    stack: list[Node] = []
    for (token_type, token_text) in tokens:
        if token_type == "LPAREN":
            node = Node(symbol=token_text, symbol_type=token_type)
            stack.append(node)
        elif token_type == "RPAREN":  # pop until '('
            node = merge_node_rparen_type(stack)
            stack.append(node)
        elif token_type == "OR":  # pop until '(' or end of stack
            or_node = Node(symbol=token_text, symbol_type=token_type)
            node = merge_node_or_type(stack)
            or_node.left = node
            stack.append(or_node)
        elif token_type in ("STAR", "OPTIONAL", "PLUS"):
            node = RepeatNode(symbol=token_text, symbol_type=token_type)
            top_node = stack.pop()
            assert top_node.symbol_type == "LITERAL", "Invalid syntax, only LITERAL type before '*' or '?'"  # noqa
            node.concat_node(top_node)
            stack.append(node)
        elif token_type == "LITERAL":
            node = Node(symbol=token_text, symbol_type=token_type)
            stack.append(node)
        elif token_type == "UNKNOWN":
            raise ValueError(f"Unknown token type {token_text}")
        else:
            raise NotImplementedError(f"Token type {token_type} not implemented")

    # reduce the stack to a single node
    return merge_stack_to_one_node(stack)


@dataclass
class State:
    cur_pattern: str = ""
    next_states: list['State'] = field(default_factory=list)

    def matched_next_states(self, text: str) -> list['State']:
        if self.match_current(text):
            return self.next_states
        return []

    def match_current(self, text: str) -> bool:
        if not text:
            return False
        return text[0] == self.cur_pattern

    @staticmethod
    def from_node(node: Node) -> State:
        state = State(cur_pattern=node.symbol)
        next_state = node_to_state(node.right) if node.right else Success()
        state.next_states = [next_state]
        return state

    def reset(self):
        pass  # default do nothing

    def to_trace(self):
        state_copy = deepcopy(self)
        state_copy.next_states = None
        return state_copy


class Success(State):

    def matched_next_states(self, text: str) -> list['State']:
        return [self]


class OrState(State):
    # state representing '|'

    def matched_next_states(self, text: str) -> list[State]:
        match_states = []
        for s in self.next_states:
            match_states.extend(s.matched_next_states(text))
        return match_states

    @staticmethod
    def from_node(node: Node) -> State:
        assert node.left is not None and node.right is not None
        left_state = node_to_state(node.left)
        right_state = node_to_state(node.right)
        state = OrState(cur_pattern=node.symbol, next_states=[left_state, right_state])
        return state


@dataclass
class Countable(State):
    # state representing {m,n} type

    min_count: int = 0
    max_count: int = 1
    count: int = 0
    single_count: bool = True
    out_loop_states: list[State] = field(default_factory=list)

    def matched_next_states(self, text: str) -> list[State]:
        if self.match_current(text):
            self.count += 1
            inloop: list[State] = [self] if self.single_count else self.next_states
            if self.count >= self.max_count:  # outloop
                return self.out_loop_states
            elif self.count < self.min_count:  # inloop
                return inloop
            else:
                return inloop + self.out_loop_states
        else:  # not match, current value should match next_state
            if self.count >= self.min_count:
                state_list = []
                for s in self.out_loop_states:
                    state_list.extend(s.matched_next_states(text))
                return state_list
            else:
                return []

    @classmethod
    def from_node(cls, node: Node) -> State:
        assert node.left is not None
        assert node.left.symbol_type == "LITERAL"
        right_state = node_to_state(node.right) if node.right else Success()
        state = cls(cur_pattern=node.left.symbol, out_loop_states=[right_state])
        if node.left.right is not None:  # not a single repeat node
            state.single_count = False
            deep_state = node_to_state(node.left.right)
            replace_success_state(deep_state, state)
            state.next_states.append(deep_state)
        return state

    def reset(self):
        self.count = 0

    def to_trace(self):
        trace = super().to_trace()
        trace.out_loop_states = None
        return trace


@dataclass
class Star(Countable):

    min_count: int = 0
    max_count: int | float = float('inf')


@dataclass
class Optional(Countable):

    min_count: int = 0
    max_count: int = 1


@dataclass
class Plus(Countable):

    min_count: int = 1
    max_count: int | float = float('inf')


def node_to_state(node: Node) -> State:
    if node.symbol_type == "STAR":
        state = Star.from_node(node)
    elif node.symbol_type == "OR":
        state = OrState.from_node(node)
    elif node.symbol_type == "OPTIONAL":
        state = Optional.from_node(node)
    elif node.symbol_type == "PLUS":
        state = Plus.from_node(node)
    else:
        state = State.from_node(node)
    return state


def regex_to_states(regex) -> State:
    syntax_tree = ast(list(tokenize(regex)))
    state = node_to_state(syntax_tree)
    return state


def replace_success_state(state: State, new_state: State):
    if isinstance(state, Success):
        return new_state

    new_list = []
    for s in state.next_states:
        if isinstance(s, Success):
            new_list.append(new_state)
        else:
            new_list.append(replace_success_state(s, new_state))
    state.next_states = new_list
    return state


def reset_state(state: State, visited=None):
    if visited is None:
        visited = set()

    visited.add(id(state))
    state.reset()
    for s in state.next_states:
        if id(s) not in visited:
            reset_state(s, visited=visited)


if __name__ == "__main__":
    regex = "(ab)*c"
    # regex = "ab*c"
    syntax_tree = ast(list(tokenize(regex)))
    print(syntax_tree)
    state = node_to_state(syntax_tree)
    print(state)
