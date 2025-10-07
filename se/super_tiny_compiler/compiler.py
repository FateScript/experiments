# a python version of https://github.com/jamiebuilds/the-super-tiny-compiler

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pprint import pprint


@dataclass
class Token:
    type: str
    value: str


def tokenizer(text) -> list[Token]:
    tokens = []
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "(" or char == ")":
            tokens.append(Token(type="paren", value=char))
            idx += 1
            continue
        elif char.isnumeric():
            value = ""
            while char.isnumeric():
                value += char
                idx += 1
                if idx >= len(text):
                    break
                char = text[idx]
            tokens.append(Token(type="number", value=value))
            continue
        elif char.isalpha():
            value = ""
            while char.isalpha():
                value += char
                idx += 1
                if idx >= len(text):
                    break
                char = text[idx]
            tokens.append(Token(type="name", value=value))
            continue
        elif char == '"' or char == "'":
            value = char
            idx += 1
            char = text[idx]
            while char != '"' and char != "'":
                value += char
                idx += 1
                if idx >= len(text):
                    raise IndexError("Unterminated string")
                char = text[idx]
            value += char
            idx += 1
            tokens.append(Token(type="string", value=value))
            continue
        elif char.isspace():
            idx += 1
            continue
    return tokens


def parser(tokens: list[Token]) -> dict:
    program = {"type": "Program", "body": []}
    idx = 0

    def walk() -> dict:
        nonlocal idx

        token = tokens[idx]
        if token.type == "paren" and token.value == "(":
            idx += 1
            token = tokens[idx]
            assert token.type == "name"
            node = {
                "type": 'CallExpression',
                "name": token.value,
                "params": [],
            }
            idx += 1
            token = tokens[idx]
            while (token.type != "paren") or (token.type == "paren" and token.value != ")"):
                node["params"].append(walk())
                token = tokens[idx]
            assert token.type == "paren" and token.value == ")"
            idx += 1
            return node
        elif token.type == "number":
            idx += 1
            return {
                "type": 'NumberLiteral',
                "value": token.value,
            }
        elif token.type == "string":
            idx += 1
            return {
                "type": 'StringLiteral',
                "value": token.value,
            }
        else:
            raise ValueError(f"Unknown token type: {token.type}, value: {token.value}")

    while idx < len(tokens):
        program["body"].append(walk())

    return program


def traverser(ast, visitor: dict):

    def traverse_array(array, parent):
        for child in array:
            traverse_node(child, parent)

    def traverse_node(node, parent):
        node_type = node["type"]
        node_func = visitor.get(node_type)
        if node_func and node_func.get("enter"):
            enter_func = node_func["enter"]
            enter_func(node, parent)

        if node_type == "Program":
            traverse_array(node["body"], node)
        elif node_type == "CallExpression":
            traverse_array(node["params"], node)
        elif node_type == "NumberLiteral" or node_type == "StringLiteral":
            pass
        else:
            raise ValueError(f"Unknown node type: {node_type}")

        if node_func and node_func.get("exit"):
            exit_func = node_func["exit"]
            exit_func(node, parent)

    traverse_node(ast, None)  # root node has no parent


def transformer(ast: dict) -> dict:
    new_ast = deepcopy(ast)

    def call_expression_enter(node, parent):
        node["callee"] = {"type": "Identifier", "name": node["name"]}
        node["arguments"] = []
        # breakpoint()
        # TODO: exit logic

    def call_expression_exit(node, parent):
        # remove useless name
        node.pop("name")
        node.pop("params")
        if parent["type"] != "CallExpression":
            cur_node = deepcopy(node)
            node["type"] = "ExpressionStatement"
            node["expression"] = cur_node
            node.pop("callee")
            node.pop("arguments")
        else:
            parent["arguments"].append(node)

    def number_string_enter(node, parent):
        parent["arguments"].append(node)

    visitor = {
        "NumberLiteral": {"enter": number_string_enter},
        "StringLiteral": {"enter": number_string_enter},
        "CallExpression": {"enter": call_expression_enter, "exit": call_expression_exit},
    }
    traverser(new_ast, visitor)

    return new_ast


def code_generator(node: dict) -> str:
    node_type = node["type"]
    if node_type == "Program":
        return "\n".join([code_generator(x) for x in node["body"]])
    elif node_type == "ExpressionStatement":
        return code_generator(node["expression"]) + ";"
    elif node_type == "CallExpression":
        func_name = code_generator(node["callee"])
        args = ", ".join([code_generator(x) for x in node["arguments"]])
        return func_name + "(" + args + ")"
    elif node_type == "Identifier":
        return node["name"]
    elif node_type == "NumberLiteral":
        return node["value"]
    elif node_type == "StringLiteral":
        return node["value"]
    else:
        raise ValueError(f"Unknown node type: {node_type}, {node}")


def compiler(source: str) -> str:
    tokens = tokenizer(source)
    ast = parser(tokens)
    new_ast = transformer(ast)
    code = code_generator(new_ast)
    return code


if __name__ == "__main__":
    text_to_tokenize = ["(add 2 (subtract 4 2))", '(concat "foo" "bar")']
    text_to_tokenize.append(text_to_tokenize[0] + text_to_tokenize[1])
    for text in text_to_tokenize:
        print(f"Tokenizing: {text}")
        tokens = tokenizer(text)
        pprint(tokens)
        ast = parser(tokens)
        pprint(ast)
        print("Transforming...")
        new_ast = transformer(ast)
        pprint(new_ast)
        code = code_generator(new_ast)
        print(f"Code:\n{code}")
        print("---" * 20)
