from typing import Any, TypeGuard, TypeVar

T = TypeVar("T")


def is_list_of_type(
    element_type: type[T],
    obj: list[Any],
) -> TypeGuard[list[T]]:
    return len(obj) > 0 and all(isinstance(elem, element_type) for elem in obj)
