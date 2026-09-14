from enum import Enum

class Occur(Enum):
    SHOULD = "SHOULD"
    MUST = "MUST"
    MUST_NOT = "MUST_NOT"

class MatchQuery:
    query: str
    column: str
    def __init__(self, query: str, column: str) -> None: ...

class BooleanQuery:
    queries: list[tuple[Occur, MatchQuery]]
    def __init__(self, queries: list[tuple[Occur, MatchQuery]]) -> None: ...
