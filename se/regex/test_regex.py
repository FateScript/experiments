import pytest

from regex_state_machine import StatesMachine


@pytest.mark.parametrize("regex, candidates, expected", [
    ("abcd", ["abcd", "abce", "acde"], [True, False, False]),
    ("ab*", ["abb", "a", "b", "abb", "abbc"], [True, True, False, True, True]),
    ("ab*c", ["abcd", "abce", "acde", "acabcde", "abbbbcd", "bbbcd"], [True, True, True, True, True, False]),
    ("a*bc", ["abcd", "abce", "acde", "acabcde", "bbbcd"], [True, True, False, True, True]),
    ("ab?c", ["abcd", "acde", "bbbcd", "abbbc"], [True, True, False, False]),
    ("ab+c", ["abcd", "acde", "abbbcd"], [True, False, True]),
])
def test_states_machine(regex, candidates, expected):
    sm = StatesMachine(regex)
    for t, exp in zip(candidates, expected):
        assert sm.match(t) == exp


@pytest.mark.parametrize("regex, candidates, expected", [
    ("(ab*c|a*bc)", ["abcd", "abce", "acde", "acabcde", "bbbcd"], [True, True, True, True, True]),
    ("(ab)+c", ["abcd", "cd", "ababad", "ababcd", "addd"], [True, False, False, True, False]),
    ("(ab)*c", ["abcd", "cd", "ababad", "ababcd", "addd"], [True, True, False, True, False]),
])
def test_complex_regex(regex, candidates, expected):
    sm = StatesMachine(regex)
    for t, exp in zip(candidates, expected):
        assert sm.match(t) == exp


@pytest.mark.parametrize("regex, candidates, count", [
    ("(ab)*c", ["abcd", "cd", "ababcd"], [1, 0, 2]),
    ("(ab)+c", ["abcd", "ababcd", "abababc"], [1, 2, 3]),
])
def test_quote_regex(regex, candidates, count):
    sm = StatesMachine(regex, trace=True)
    for t, cnt in zip(candidates, count):
        sm.match(t)
        max_count = 0
        path = sm.find_success_path()
        for state in path:
            if hasattr(state, 'count') and state.count > max_count:
                max_count = state.count
        assert max_count == cnt
