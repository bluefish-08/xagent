"""Tests for the shared (app-injected) MCP env hook and layered merge."""

from xagent.web.services import mcp_runtime


def test_merge_stdio_env_layers_later_wins():
    """global -> shared -> user, each layer overriding the previous."""
    merged = mcp_runtime.merge_stdio_env(
        {"A": "g", "B": "g", "C": "g"},
        {"B": "s", "C": "s"},
        {"C": "u"},
    )
    assert merged == {"A": "g", "B": "s", "C": "u"}


def test_merge_stdio_env_two_arg_unchanged():
    """Existing two-arg callers keep working (user over global)."""
    assert mcp_runtime.merge_stdio_env({"A": "g"}, {"A": "u"}) == {"A": "u"}
    assert mcp_runtime.merge_stdio_env({"A": "g"}, None) == {"A": "g"}


def test_shared_env_hook_default_is_noop():
    mcp_runtime.set_mcp_shared_env_hook(None)
    assert mcp_runtime.load_shared_env_overrides(object(), 1) == {}


def test_shared_env_hook_is_invoked():
    calls = {}

    def hook(db, user_id):
        calls["args"] = (db, user_id)
        return {5: {"KEY": "shared-value"}}

    mcp_runtime.set_mcp_shared_env_hook(hook)
    try:
        assert mcp_runtime.load_shared_env_overrides("db", 7) == {
            5: {"KEY": "shared-value"}
        }
        assert calls["args"] == ("db", 7)
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)


def test_shared_env_hook_none_user_returns_empty():
    mcp_runtime.set_mcp_shared_env_hook(lambda db, uid: {1: {"K": "v"}})
    try:
        assert mcp_runtime.load_shared_env_overrides("db", None) == {}
    finally:
        mcp_runtime.set_mcp_shared_env_hook(None)
