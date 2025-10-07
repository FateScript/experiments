#!/usr/bin/env python3

import pytest

from compiler import Token, tokenizer, parser, transformer, compiler


@pytest.mark.parametrize("text, expected", [
    (
        "(add 2 (subtract 4 2))",
        [
            Token(type="paren", value="("),
            Token(type="name", value="add"),
            Token(type="number", value="2"),
            Token(type="paren", value="("),
            Token(type="name", value="subtract"),
            Token(type="number", value="4"),
            Token(type="number", value="2"),
            Token(type="paren", value=")"),
            Token(type="paren", value=")")
        ]
    ),
    (
        '(concat "foo" "bar")',
        [
            Token(type="paren", value="("),
            Token(type="name", value="concat"),
            Token(type="string", value='"foo"'),
            Token(type="string", value='"bar"'),
            Token(type="paren", value=")")
        ]
    ),
])
def test_tokenizer(text, expected):
    tokens = tokenizer(text)
    assert tokens == expected


@pytest.mark.parametrize("text, expected", [
    (
        "(add 2 (subtract 4 2))",
        {'body': [{
            'name': 'add',
            'params': [
                {'type': 'NumberLiteral', 'value': '2'},
                {
                    'name': 'subtract',
                    'params': [
                        {'type': 'NumberLiteral', 'value': '4'},
                        {'type': 'NumberLiteral', 'value': '2'}
                    ],
                    'type': 'CallExpression'
                },
            ],
            'type': 'CallExpression'
        }],
         'type': 'Program'}
    ),
])
def test_parser(text, expected):
    tokens = tokenizer
    tokens = tokenizer(text)
    ast = parser(tokens)
    assert ast == expected


@pytest.mark.parametrize("text, expected", [
    (
        "(add 2 (subtract 4 2))",
        {'body': [{
            'type': 'ExpressionStatement',
            'expression': {
                'type': 'CallExpression',
                'callee': {'name': 'add', 'type': 'Identifier'},
                'arguments': [
                    {'type': 'NumberLiteral', 'value': '2'},
                    {
                        'type': 'CallExpression',
                        'callee': {'name': 'subtract', 'type': 'Identifier'},
                        'arguments': [
                            {'type': 'NumberLiteral', 'value': '4'},
                            {'type': 'NumberLiteral', 'value': '2'}
                        ],
                    }
                ],
            }
        }],
         'type': 'Program'},
    ),
])
def test_transformer(text, expected):
    tokens = tokenizer
    tokens = tokenizer(text)
    ast = parser(tokens)
    transformed_ast = transformer(ast)
    assert transformed_ast == expected


@pytest.mark.parametrize("text, expected", [
    ("(add 2 (subtract 4 2))", "add(2, subtract(4, 2));"),
    ('(concat "foo" "bar")', 'concat("foo", "bar");'),
    (
        '(add 2 (subtract 4 2)) (concat "foo" "bar")',
        'add(2, subtract(4, 2));\nconcat("foo", "bar");'
    ),
])
def test_compiler(text, expected):
    output = compiler(text)
    assert output == expected
