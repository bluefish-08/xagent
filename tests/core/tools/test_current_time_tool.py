from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from xagent.core.tools.adapters.vibe import current_time_tool as module
from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.current_time_tool import (
    CurrentTimeTool,
    current_time,
)

# The instant behind the reported incident: already the next day in Melbourne.
FROZEN = datetime(2026, 8, 24, 22, 3, 37, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_now", lambda: FROZEN)


def test_reports_utc_when_no_zone_is_supplied() -> None:
    result = current_time()

    assert result.utc == "2026-08-24 22:03:37"
    assert result.local == result.utc
    assert result.timezone == "UTC"
    assert result.utc_offset == "+00:00"


def test_reports_local_time_for_a_supplied_zone() -> None:
    result = current_time("Australia/Melbourne")

    assert result.utc == "2026-08-24 22:03:37"
    assert result.local == "2026-08-25 08:03:37"
    assert result.timezone == "Australia/Melbourne"
    assert result.utc_offset == "+10:00"


def test_reports_a_half_hour_offset() -> None:
    result = current_time("Asia/Kolkata")

    assert result.local == "2026-08-25 03:33:37"
    assert result.utc_offset == "+05:30"


def test_reports_a_negative_offset() -> None:
    result = current_time("America/New_York")

    assert result.local == "2026-08-24 18:03:37"
    assert result.utc_offset == "-04:00"


def test_follows_daylight_saving_for_the_same_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module, "_now", lambda: datetime(2026, 12, 24, 22, 3, 37, tzinfo=timezone.utc)
    )

    result = current_time("Australia/Melbourne")

    assert result.local == "2026-12-25 09:03:37"
    assert result.utc_offset == "+11:00"


@pytest.mark.parametrize(
    "supplied",
    [
        "Not/AZone",
        "",
        "   ",
        "Australia/Melbourne\x00",
        "A" * 5000,
        "../../../etc/passwd",
        "/etc/passwd",
        None,
    ],
)
def test_unusable_zone_degrades_to_utc(supplied: object) -> None:
    result = current_time(supplied)  # type: ignore[arg-type]

    assert result.timezone == "UTC"
    assert result.local == result.utc == "2026-08-24 22:03:37"


def test_tool_runs_through_the_json_surface() -> None:
    tool = CurrentTimeTool()

    assert tool.run_json_sync({"timezone": "Australia/Melbourne"}) == {
        "utc": "2026-08-24 22:03:37",
        "local": "2026-08-25 08:03:37",
        "timezone": "Australia/Melbourne",
        "utc_offset": "+10:00",
    }
    assert asyncio.run(tool.run_json_async({})) == {
        "utc": "2026-08-24 22:03:37",
        "local": "2026-08-24 22:03:37",
        "timezone": "UTC",
        "utc_offset": "+00:00",
    }


def test_tool_declares_a_read_only_basic_identity() -> None:
    metadata = CurrentTimeTool().metadata

    assert metadata.name == "get_current_time"
    assert metadata.category == ToolCategory.BASIC
    # Reading a clock touches nothing, so the ReAct scheduler may run it
    # alongside other concurrency-safe tools.
    assert metadata.read_only is True
    assert metadata.concurrency_safe is True


def test_description_tells_the_model_the_prompt_clock_is_stale() -> None:
    description = CurrentTimeTool().description

    assert "does not advance" in description
    assert "timezone" in description


def test_module_is_imported_by_the_tool_registry() -> None:
    """Without this import the @register_tool decorator never runs."""
    from xagent.core.tools.adapters.vibe import factory

    source = factory.__loader__.get_source(factory.__name__)  # type: ignore[union-attr]
    assert "current_time_tool," in source
