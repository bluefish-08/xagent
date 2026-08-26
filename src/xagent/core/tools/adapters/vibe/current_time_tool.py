from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Type
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from .....web.tools.config import WebToolConfig
from .base import AbstractBaseTool, ToolCategory, ToolVisibility
from .factory import register_tool

_STAMP = "%Y-%m-%d %H:%M:%S"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CurrentTimeArgs(BaseModel):
    timezone: Optional[str] = Field(
        default=None,
        description=(
            "IANA timezone name to report local time in, for example "
            "'Australia/Melbourne'. Copy it from the zone named in the system "
            "prompt's date-and-time line. Omit it when that line reports UTC "
            "only. An unresolvable name falls back to UTC, and the 'timezone' "
            "field of the result says which zone was actually used."
        ),
    )


class CurrentTimeResult(BaseModel):
    utc: str = Field(description="Current UTC time, as YYYY-MM-DD HH:MM:SS.")
    local: str = Field(
        description=(
            "Current time in the reported zone, as YYYY-MM-DD HH:MM:SS. Equal "
            "to 'utc' when the zone is UTC."
        )
    )
    timezone: str = Field(
        description=(
            "Zone the 'local' field is expressed in. 'UTC' when none was "
            "supplied or the supplied name could not be resolved."
        )
    )
    utc_offset: str = Field(
        description="Offset of the reported zone from UTC, for example '+10:00'."
    )


def _resolve_zone(name: Optional[str]) -> Optional[ZoneInfo]:
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return ZoneInfo(name.strip())
    except (ZoneInfoNotFoundError, ValueError, OSError, TypeError, KeyError):
        # The name comes from the model, so an unusable one degrades to UTC
        # rather than failing the call. OSError covers an over-long name.
        return None


def _format_offset(offset: timedelta) -> str:
    sign = "-" if offset < timedelta(0) else "+"
    hours, remainder = divmod(abs(offset), timedelta(hours=1))
    minutes = remainder // timedelta(minutes=1)
    return f"{sign}{hours:02d}:{minutes:02d}"


def current_time(timezone_name: Optional[str] = None) -> CurrentTimeResult:
    """Read the wall clock now, in UTC and optionally in a named zone."""
    now_utc = _now()
    zone = _resolve_zone(timezone_name)
    local = now_utc.astimezone(zone) if zone is not None else now_utc
    return CurrentTimeResult(
        utc=now_utc.strftime(_STAMP),
        local=local.strftime(_STAMP),
        timezone=zone.key if zone is not None else "UTC",
        utc_offset=_format_offset(local.utcoffset() or timedelta(0)),
    )


class CurrentTimeTool(AbstractBaseTool):
    """Answers 'what time is it now', which the system prompt cannot."""

    category = ToolCategory.BASIC
    read_only = True  # reads a clock ⇒ concurrency-safe

    def __init__(self) -> None:
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return (
            "Return the real current time. The date and time in the system "
            "prompt is stamped once when the turn begins and does not advance "
            "while the turn runs, so call this whenever the answer depends on "
            "the time now rather than on when the turn started: measuring how "
            "long something took, checking whether a deadline has passed, or "
            "resolving a relative date during a turn that may have crossed "
            "midnight. Pass the timezone named in the system prompt's "
            "date-and-time line to get local time alongside UTC."
        )

    def args_type(self) -> Type[BaseModel]:
        return CurrentTimeArgs

    def return_type(self) -> Type[BaseModel]:
        return CurrentTimeResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        parsed = CurrentTimeArgs.model_validate(args)
        return current_time(parsed.timezone).model_dump()

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        return self.run_json_sync(args)


@register_tool(selection_gate="intrinsic")
async def create_current_time_tool(config: WebToolConfig) -> list[AbstractBaseTool]:
    """Create the current-time tool."""
    return [CurrentTimeTool()]
