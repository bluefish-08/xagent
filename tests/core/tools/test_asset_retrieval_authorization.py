"""Asset-retrieval tool descriptions must not authorize unprompted retrieval.

These descriptions are injected into the tool schema on every tool-capable step,
while a skill's policy is only present once it has been loaded. A description
that offers a wider route than the skill allows therefore wins by default, so the
authorization wording has to live here too.

The surfaces are read from the tools as they are actually built, and the
inventory is walked out of the adapter package, so a new retrieval tool fails
here rather than slipping past a hand-written list.
"""

import importlib
import pkgutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

import xagent.core.tools.adapters.vibe as vibe_pkg
from xagent.core.model.image.base import BaseImageModel
from xagent.core.tools.adapters.vibe.base import AbstractBaseTool, ToolCategory
from xagent.core.tools.adapters.vibe.download_web_asset import (
    DownloadWebAssetArgs,
    DownloadWebAssetTool,
)
from xagent.core.tools.adapters.vibe.fetch_web_content import (
    FetchWebContentArgs,
    FetchWebContentTool,
)
from xagent.core.tools.adapters.vibe.image_tool import ImageGenerationTool
from xagent.skills.parser import SkillParser

SKILL_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "xagent"
    / "skills"
    / "builtin"
    / "static-visual-design"
)


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _field(model: type[BaseModel], name: str) -> str:
    return _normalized(model.model_fields[name].description or "")


def _built_image_descriptions() -> dict[str, str]:
    """The generate/edit descriptions as `get_tools` emits them to the schema.

    Reading the class constants instead would miss anything the builder prepends
    or interpolates.
    """
    model = Mock(spec=BaseImageModel)
    workspace = Mock()
    workspace.output_dir = Path("/tmp/asset-authorization-test")
    tool = ImageGenerationTool(
        {"m": model},
        {"m": "test model"},
        workspace,
        default_generate_model=model,
        default_edit_model=model,
    )
    return {t.name: _normalized(t.description) for t in tool.get_tools()}


def _surfaces() -> dict[str, str]:
    """Every description reaching the model alongside an asset-fetch route."""
    image = _built_image_descriptions()
    return {
        "fetch_web_content.description": _normalized(FetchWebContentTool().description),
        "fetch_web_content.include_assets": _field(
            FetchWebContentArgs, "include_assets"
        ),
        "download_web_asset.description": _normalized(
            DownloadWebAssetTool(None).description  # type: ignore[arg-type]
        ),
        "download_web_asset.url": _field(DownloadWebAssetArgs, "url"),
        "generate_image.description": image["generate_image"],
        "edit_image.description": image["edit_image"],
    }


# Tools that can pull remote content into the run and therefore need the
# authorization wording asserted below.
RETRIEVAL_TOOL_CLASSES = {
    "FetchWebContentTool": "fetch_web_content",
    "DownloadWebAssetTool": "download_web_asset",
}

# Tools in the same categories that cannot: query-only search returns text and
# takes no URL, and the image FunctionTool wrapper carries whichever description
# the builder hands it, which is asserted through the built surfaces instead.
NOT_A_RETRIEVAL_ROUTE = {
    "ExaWebSearchTool",
    "TavilyWebSearchTool",
    "WebSearchTool",
    "ZhipuWebSearchTool",
    "ImageGenerationFunctionTool",
}


def _discovered_tool_classes() -> dict[str, type]:
    """Every tool class in the adapter package, imported for its side effect."""
    for module in pkgutil.iter_modules(vibe_pkg.__path__):
        importlib.import_module(f"{vibe_pkg.__name__}.{module.name}")

    def descendants(cls: type) -> Iterator[type]:
        for sub in cls.__subclasses__():
            yield sub
            yield from descendants(sub)

    return {
        c.__name__: c
        for c in descendants(AbstractBaseTool)
        if getattr(c, "category", None) in (ToolCategory.WEB_SEARCH, ToolCategory.IMAGE)
    }


BANNED_WORDING = (
    "when looking for logos",
    "usually asset_query='logo'",
    "prefer the official brand domain",
    "retrieved from the brand's own official source",
    "discovers an official logo",
    "do not go looking",
)

# One exact phrase per surface. An aggregate count passes on a coincidental
# "directly" elsewhere in an unrelated sentence.
REQUIRED_WORDING = {
    "fetch_web_content.description": (
        "when the user asked you to obtain an exact asset"
    ),
    "fetch_web_content.include_assets": (
        "Enable it when the user asked you to inspect or enumerate what a page loads"
    ),
    "download_web_asset.description": (
        "The user has to have asked you to obtain this asset"
    ),
    "download_web_asset.url": "for an asset the user asked you to obtain",
    "generate_image.description": "an asset the user directed you to retrieve",
    "edit_image.description": "an asset the user directed you to retrieve",
}


def test_every_retrieval_tool_is_classified() -> None:
    """A new web or image tool has to be triaged, not silently uncovered."""
    discovered = set(_discovered_tool_classes())
    classified = set(RETRIEVAL_TOOL_CLASSES) | NOT_A_RETRIEVAL_ROUTE

    assert discovered <= classified, (
        f"unclassified web/image tools: {sorted(discovered - classified)}. Add each "
        f"to RETRIEVAL_TOOL_CLASSES with its authorization wording, or to "
        f"NOT_A_RETRIEVAL_ROUTE if it cannot pull remote content in."
    )
    assert set(RETRIEVAL_TOOL_CLASSES) <= discovered, (
        f"renamed or removed: {sorted(set(RETRIEVAL_TOOL_CLASSES) - discovered)}"
    )


def test_every_retrieval_tool_has_a_covered_surface() -> None:
    """Each retrieval tool contributes at least one asserted surface."""
    covered = {name.split(".")[0] for name in REQUIRED_WORDING}
    for cls_name, tool_name in RETRIEVAL_TOOL_CLASSES.items():
        assert tool_name in covered, f"{cls_name} ({tool_name}) has no asserted surface"


def test_every_surface_has_required_wording() -> None:
    assert set(_surfaces()) == set(REQUIRED_WORDING), (
        "a retrieval surface was added or renamed without its authorization wording"
    )


@pytest.mark.parametrize("name", sorted(_surfaces()))
def test_surface_forbids_unprompted_retrieval(name: str) -> None:
    lowered = _surfaces()[name].lower()
    for phrase in BANNED_WORDING:
        assert phrase not in lowered, f"{name} reintroduced: {phrase!r}"


@pytest.mark.parametrize("name,phrase", sorted(REQUIRED_WORDING.items()))
def test_surface_requires_user_authorization(name: str, phrase: str) -> None:
    assert phrase in _surfaces()[name], (
        f"{name} no longer conditions retrieval on the user asking: {phrase!r}"
    )


def test_authorization_follows_the_request_not_the_url() -> None:
    """A page web_search found for a retrieval the user asked for is in scope."""
    fetch = _surfaces()["fetch_web_content.description"]

    assert "one they named or one web_search found for that request" in fetch
    assert "uninstructed" in fetch


def test_a_user_supplied_url_is_a_route_but_not_a_licence() -> None:
    """A URL supplied with an obtain request is a valid route; one pasted to be
    read is not. The obtain intent has to qualify both routes, not only the
    surfaced one.
    """
    surfaces = _surfaces()

    assert "they supplied with that request" in surfaces["download_web_asset.url"]
    assert "pasted to be read or discussed" in surfaces["download_web_asset.url"]
    assert (
        "use a URL they supplied directly"
        in (surfaces["download_web_asset.description"])
    )


def test_enumeration_does_not_authorize_a_download() -> None:
    """include_assets is inspect-only, so a URL it lists is not yet acquirable."""
    surfaces = _surfaces()

    assert (
        "does not authorize download_web_asset"
        in (surfaces["fetch_web_content.include_assets"])
    )
    assert "does not authorize a download" in surfaces["download_web_asset.description"]
    assert "merely listed, are not such requests" in surfaces["download_web_asset.url"]
    # The handoff sentence sits right after the inspect-only branch, so it has to
    # carry the condition rather than apply to every URL that comes back.
    fetch = surfaces["fetch_web_content.description"]
    assert "when the user asked you to obtain one, pass the chosen URL" in fetch
    assert "being asked to inspect a page is not such a request" in fetch


def test_include_assets_is_also_an_enumeration_contract() -> None:
    """The flag lists page resources without downloading them."""
    flag = _surfaces()["fetch_web_content.include_assets"]

    assert "inspect or enumerate" in flag
    assert "nothing is downloaded" in flag
    assert "asset-hunting on your own" in flag


def test_page_wide_inspection_leaves_asset_query_empty() -> None:
    """`_filter_and_deduplicate_assets` only filters on a non-empty query.

    Telling the model to always set asset_query would drop exactly the unrelated
    assets a page-wide inspection is asking for.
    """
    assert (
        "leaving asset_query empty so nothing is filtered out"
        in (_surfaces()["fetch_web_content.description"])
    )
    assert "Leave it empty to list everything the page loads" in (
        _field(FetchWebContentArgs, "asset_query")
    )


def test_authenticity_is_independent_of_authorization() -> None:
    """A requested retrieval can still return a counterfeit."""
    surfaces = _surfaces()

    assert "however the URL was reached" in surfaces["download_web_asset.description"]
    assert (
        "only when the user supplied it or confirmed the source"
        in (surfaces["download_web_asset.description"])
    )
    assert "unverified until the user confirms it" in surfaces["download_web_asset.url"]


def test_asking_the_user_stops_once_they_have_chosen() -> None:
    """An unscoped "ask the user" reopens a question the user already answered."""
    for name in ("generate_image.description", "edit_image.description"):
        text = _surfaces()[name]
        assert "ask the user how to proceed" in text, name
        assert "act on that choice instead of asking again" in text, name


def test_skill_and_tool_descriptions_agree_on_retrieval() -> None:
    """Both land in the same context once the skill is loaded.

    A skill that sanctions self-directed retrieval overrides the tool schema's
    prohibition, so the two contracts have to move together.
    """
    body = " ".join(SkillParser.parse(SKILL_DIR)["content"].split())

    assert "Two sources need no permission" in body
    assert "official web presence" not in body, (
        "the skill sanctions retrieval the tool descriptions forbid"
    )
    assert "take it only when they tell you to" in body

    surfaces = _surfaces()
    for name in ("generate_image.description", "edit_image.description"):
        assert "an asset the user directed you to retrieve" in surfaces[name], name


def test_skill_separates_looking_from_taking() -> None:
    """The skill authorizes an external look; that is not authorization to take."""
    body = " ".join(SkillParser.parse(SKILL_DIR)["content"].split())

    assert "Being told to look at a page is not being told to take what it holds" in (
        body
    )
    assert "ask before acquiring or using it" in body
