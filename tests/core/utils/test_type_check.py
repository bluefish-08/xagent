from xagent.core.utils.type_check import is_list_of_type


def test_empty_list():
    assert not is_list_of_type(str, [])
    assert not is_list_of_type(int, [])


def test_list_of_strings():
    assert is_list_of_type(str, ["a", "b"])
    assert not is_list_of_type(int, ["a", "b"])


def test_list_of_ints():
    assert is_list_of_type(int, [1, 2, 3])
    assert not is_list_of_type(str, [1, 2, 3])


def test_list_of_floats():
    assert is_list_of_type(float, [1.0, 2.0, 3.0])
    assert not is_list_of_type(int, [1.0, 2.0, 3.0])


def test_mixed_list():
    assert not is_list_of_type(int, [1, "a"])
    assert not is_list_of_type(int, ["a", "1"])


def test_single_element():
    assert is_list_of_type(int, [42])
    assert is_list_of_type(str, ["hello"])


def test_custom_class():
    class MyClass:
        pass

    obj = MyClass()
    assert is_list_of_type(MyClass, [obj])
    assert not is_list_of_type(str, [obj])
