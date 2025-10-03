#!/usr/bin/env python3

from copy import deepcopy

from regex_ast import State, Success, regex_to_states


class StatesMachine:

    def __init__(self, regex: str, trace: bool = False):
        self.regex = regex
        self.start_node = regex_to_states(regex)
        self.candidate_states = []
        self.trace = trace
        if self.trace:
            self.trace_tree = []

    def reset_state(self):
        # reset_state(self.start_node)  # deepcopy make it useless
        self.candidate_states = []

    def reach_success_state(self) -> bool:
        return any([isinstance(x, Success) for x in self.candidate_states])

    def match(self, text: str) -> bool:
        self.reset_state()
        start_node_copy = deepcopy(self.start_node)  # make start node state-less
        idx = 0
        while idx < len(text) + 1:  # + 1 to handle last string, for example, ab* match a
            choose_states = []
            trace = []
            for state in self.candidate_states + [deepcopy(start_node_copy)]:
                state: State
                next_states = state.matched_next_states(text[idx:])
                if self.trace:
                    state_trace = state.to_trace()
                    for ns in next_states:
                        trace.append((state_trace, ns.to_trace()))
                choose_states.extend(next_states)
            if self.trace:
                self.trace_tree.append(trace)
            self.candidate_states = choose_states
            if self.reach_success_state():
                return True
            idx += 1
        return self.reach_success_state()

    def find_success_path(self):
        if not self.trace:
            raise ValueError("Trace is not enabled")
        success_path = [Success().to_trace()]
        for tree in reversed(self.trace_tree):
            for state, next_state in tree:
                if next_state == success_path[-1]:
                    success_path.append(state)
                    break
        return success_path


if __name__ == "__main__":
    # Example usage
    regex = "(ab*c|a*bc)"
    sm = StatesMachine(regex)
    candidates = ["abcd", "abce", "acde", "acabcde", "bbbcd"]
    for t in candidates:
        if sm.match(t):
            print(f"{t} match the regex {sm.regex}")
