# coding: utf-8
from __future__ import unicode_literals

from copy import deepcopy, copy
from threading import Lock
from intervaltree import IntervalTree

from yargy.tokenizer import Token, Tokenizer
from yargy.normalization import NormalizationType
from yargy.utils import get_tokens_position


def create_or_copy_grammar(grammar, name=None):
    if isinstance(grammar, list):
        grammar = Grammar(name, deepcopy(grammar))
    elif isinstance(grammar, (Operation, Grammar)):
        grammar = deepcopy(grammar)
    else:
        raise ValueError('Not supported grammar type: {}'.format(grammar))
    return grammar


def build_grammars_from_multiple_classes(classes):
    for _class in classes:
        _class_name = _class.__name__
        for rule in _class.__members__.values():
            name = '{0}__{1}'.format(_class_name, rule.name)
            yield rule, name, create_or_copy_grammar(rule.value, name)


class Stack(list):

    '''
    Special list for grammar, which holds matches with rule index
    (by which token was captured)
    '''

    def have_matches_by_rule_index(self, rule_index):
        '''
        Checks that stack contains matches by rule index
        '''
        return any(
            (rule == rule_index for (rule, _) in reversed(self))
        )

    def flatten(self):
        '''
        Returns matched tokens without rule indexes
        '''
        return [value for (_, value) in self]


class Operation(object):

    def __init__(self, *grammars):
        self.grammars = grammars


class OR(Operation):

    pass


class Leaf(object):

    def __init__(self, index=0, stack=None):
        self.index = index
        self.stack = stack or Stack()

    def __repr__(self):
        return 'Leaf(index={0}, stack={1})'.format(self.index, self.stack.flatten())


class Grammar(object):

    def __init__(self, name, grammars):
        self.name = name
        self.grammars = grammars + [
            {
                'terminal': True,
            }
        ]
        self.terminal_rule_index = len(self.grammars) - 1
        self.reset()

    def shift(self, token, leafs=None):

        if not leafs:
            self.leafs.append(Leaf())

        for index, leaf in enumerate(leafs or self.leafs):

            rule = self.grammars[leaf.index]

            repeatable = rule.get('repeatable', False)
            optional = rule.get('optional', False)
            terminal = rule.get('terminal', False)
            labels = rule.get('labels', [])

            token = copy(token)

            if all(self.match(token, leaf.stack, labels)):

                if not terminal:
                    leaf.stack.append((leaf.index, token))

                if not repeatable and not terminal:
                    leaf.index += 1
            else:
                if (repeatable and leaf.stack.have_matches_by_rule_index(leaf.index)) or optional:
                    leaf.index += 1
                    return self.shift(token, leafs=[leaf])
                else:
                    del self.leafs[index]

    def reduce(self):

        for index, leaf in enumerate(self.leafs):

            # print(leaf, self.terminal_rule_index)

            if leaf.index == self.terminal_rule_index:

                del self.leafs[index]

                yield leaf.stack

    def reset(self):
        self.leafs = []

    def match(self, token, stack, labels):
        stack = stack.flatten()
        for label in labels:
            yield label(token, stack)


class Parser(object):

    '''
    Yet another GLR-parser.
    '''

    def __init__(self, grammars, tokenizer=None, pipelines=None, cache_size=0):
        self.grammars = grammars
        self.tokenizer = tokenizer or Tokenizer(cache_size=cache_size)
        self.pipelines = pipelines or []
        self.lock = Lock()

    def extract(self, text, return_flatten_stack=True):
        with self.lock:
            stream = self.tokenizer.transform(text)

            for pipeline in self.pipelines:
                stream = pipeline(stream)

            for token in stream:
                for grammar in self.grammars:
                    grammar.shift(token)

                    matches = grammar.reduce()

                    for match in matches:
                        yield (grammar, match.flatten())

            for grammar in self.grammars:
                grammar.reset()


class Combinator(object):

    '''
    Combinator merges multiple grammars (in multiple enums) into one parser
    '''

    def __init__(self, classes, *args, **kwargs):
        self.classes = {}
        self.grammars = []
        for rule, name, grammar in build_grammars_from_multiple_classes(classes):
            self.classes[name] = rule
            self.grammars.append(grammar)
        self.parser = Parser(self.grammars, *args, **kwargs)

    def extract(self, text):
        for (rule, match) in self.parser.extract(text):
            yield self.classes[rule.name], match

    def resolve_matches(self, matches, strict=True):
        # sort matches by tokens count in decreasing order
        matches = sorted(matches, key=lambda m: len(m[1]), reverse=True)
        tree = IntervalTree()
        for (grammar, match) in matches:
            start, stop = get_tokens_position(match)
            exists = tree[start:stop]
            if exists and not strict:
                for interval in exists:
                    exists_grammar, _ = interval.data
                    exists_contains_current_grammar = (
                        interval.begin < start and interval.end > stop)
                    exists_grammar_with_same_type = isinstance(
                        exists_grammar, grammar.__class__)
                    if not exists_grammar_with_same_type and exists_contains_current_grammar:
                        exists = False
            if not exists:
                tree[start:stop] = (grammar, match)
                yield (grammar, match)
