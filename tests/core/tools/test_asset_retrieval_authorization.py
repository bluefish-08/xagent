"""Asset-retrieval tool descriptions must not authorize unprompted retrieval.

A tool description is in the schema on every step; a skill's policy only after
it loads, so a wider description silently wins. Surfaces are read from the built
tools and the inventory is walked out of the package, not hand-listed.
"""

import importlib
import pkgutil
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

import xagent.core.tools.adapters.vibe as vibe_pkg
from tests.shared.fake_skill_manager import FakeSkillManager
from xagent.core.model.image.base import BaseImageModel
from xagent.core.model.video.base import BaseVideoModel
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
from xagent.core.tools.adapters.vibe.video_tool import VideoGenerationTool
from xagent.skills.manager import SkillManager
from xagent.skills.parser import SkillParser

SKILL_DIR = SkillManager.get_builtin_root() / "static-visual-design"


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _field(model: type[BaseModel], name: str) -> str:
    return _normalized(model.model_fields[name].description or "")


def _built_image_descriptions() -> dict[str, str]:
    """As `get_tools` emits them: the class constants miss builder interpolation."""
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


def _built_video_descriptions() -> dict[str, str]:
    """generate_video accepts reference_image_urls, so it is a retrieval surface."""
    model = Mock(spec=BaseVideoModel)
    workspace = Mock()
    workspace.output_dir = Path("/tmp/asset-authorization-test")
    tool = VideoGenerationTool(
        {"m": model}, {"m": "test model"}, workspace, default_video_model=model
    )
    return {t.name: _normalized(t.description) for t in tool.get_tools()}


def _built_description(builder: Callable[[], dict[str, str]], name: str) -> str:
    built = builder()
    assert name in built, f"builder no longer emits {name!r}: {sorted(built)}"
    return built[name]


# Emitted by the image builder but carrying no retrieval route: it only lists
# configured models. Named so a third tool from the same wrapper fails below.
IMAGE_TOOLS_WITHOUT_A_RETRIEVAL_ROUTE = {"list_image_models", "list_video_models"}


def _surfaces() -> dict[str, str]:
    """Every description reaching the model alongside an asset-fetch route."""
    return {
        "fetch_web_content.description": _normalized(FetchWebContentTool().description),
        "fetch_web_content.include_assets": _field(
            FetchWebContentArgs, "include_assets"
        ),
        # Carries the one remaining 'logo' example this PR removed; without it
        # here, no scan covers the field.
        "fetch_web_content.asset_query": _field(FetchWebContentArgs, "asset_query"),
        "download_web_asset.description": _normalized(
            DownloadWebAssetTool(None).description  # type: ignore[arg-type]
        ),
        "download_web_asset.url": _field(DownloadWebAssetArgs, "url"),
        "generate_image.description": _built_description(
            _built_image_descriptions, "generate_image"
        ),
        "edit_image.description": _built_description(
            _built_image_descriptions, "edit_image"
        ),
        "generate_video.description": _built_description(
            _built_video_descriptions, "generate_video"
        ),
    }


# Tools that fetch a remote asset for reuse as a reference, which is what this
# contract governs.
RETRIEVAL_TOOL_CLASSES = {
    "FetchWebContentTool": "fetch_web_content",
    "DownloadWebAssetTool": "download_web_asset",
}

# Reach the network or take a URL, but cannot turn one into a reusable asset
# reference.
NOT_A_RETRIEVAL_ROUTE = {
    # Query-only: returns text, takes no URL.
    "ExaWebSearchTool",
    "TavilyWebSearchTool",
    "WebSearchTool",
    "ZhipuWebSearchTool",
    # Wrappers whose builders emit no URL-taking parameter.
    "VisionFunctionTool",
    "AudioFunctionTool",
    "MusicFunctionTool",
    "SoundEffectFunctionTool",
    # Browser session actions on an already-open page.
    "BrowserClickTool",
    "BrowserCloseTool",
    "BrowserEvaluateTool",
    "BrowserExtractTextTool",
    "BrowserFillTool",
    "BrowserListSessionsTool",
    "BrowserPdfTool",
    "BrowserScreenshotTool",
    "BrowserSelectOptionTool",
    "BrowserWaitForSelectorTool",
    # Local file, document, and data operations.
    "FileAnalysisTool",
    "FileTool",
    "PPTXTool",
    "SQLQueryFunctionTool",
    "SkillTool",
    # Delegating wrappers: the wrapped class carries the classification.
    "OutputFilteredToolWrapper",
    "SandboxedToolWrapper",
}

# General-purpose remote routes predating this contract. Listed as an explicit
# carve-out rather than a silent omission; tightening them is its own change.
GENERIC_REMOTE_ROUTE_OUT_OF_SCOPE = {
    # Builder-produced: governed instances are asserted as built surfaces.
    "FunctionTool",
    "ImageGenerationFunctionTool",
    "VideoGenerationFunctionTool",
    "APITool",
    "CustomApiTool",
    "MCPToolAdapter",
    "BrowserNavigateTool",
    "ComputerTool",
    "CreateKnowledgeBaseFromUrlTool",
    # Code executors fetch anything by construction.
    "CommandExecutorFunctionTool",
    "JavaScriptExecutorFunctionTool",
    "PythonExecutorFunctionTool",
    "_SshToolBase",
}

# Field names that make a tool capable of pulling a caller-named remote resource.
URL_BEARING_FIELDS = {
    "url",
    "urls",
    "image_url",
    "image_urls",
    "images",
    "src",
    "source_url",
}

# Categories whose tools reach outside the run by construction, even when no
# field is named like a URL.
REMOTE_CAPABLE_CATEGORIES = (
    ToolCategory.WEB_SEARCH,
    ToolCategory.IMAGE,
    ToolCategory.VISION,
    ToolCategory.BROWSER,
)


def _discovered_tool_classes() -> dict[str, type]:
    """Every tool class that can pull a remote resource into the run."""
    # walk_packages, not iter_modules: subpackages such as sandboxed_tool are
    # only imported lazily by the factory and would otherwise stay invisible.
    for module in pkgutil.walk_packages(vibe_pkg.__path__, f"{vibe_pkg.__name__}."):
        importlib.import_module(module.name)

    def descendants(cls: type) -> Iterator[type]:
        for sub in cls.__subclasses__():
            yield sub
            yield from descendants(sub)

    def remote_capable(cls: type) -> bool:
        category = getattr(cls, "category", None)
        if isinstance(category, property):
            # Declared as a property, so its value needs an instance; treat it
            # as unknown rather than silently not-remote.
            return True
        if category in REMOTE_CAPABLE_CATEGORIES:
            return True
        try:
            # Unbound on purpose: most args_type implementations ignore self.
            fields = set(cls.args_type(cls).model_fields)  # type: ignore[attr-defined]
        except Exception:
            # Not statically introspectable; a swallowed error must never read
            # as "not remote-capable".
            return True
        return bool(URL_BEARING_FIELDS & fields)

    return {
        c.__name__: c
        for c in set(descendants(AbstractBaseTool))
        # __subclasses__ is process-global: test fixtures subclass this too.
        if c.__module__.startswith(vibe_pkg.__name__) and remote_capable(c)
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
        "Enable it when the user asked you to inspect or enumerate the static "
        "references a page declares"
    ),
    "fetch_web_content.asset_query": "Leave it empty to list every supported static",
    "download_web_asset.description": (
        "The user has to have asked you to obtain this asset"
    ),
    "download_web_asset.url": "for an asset the user asked you to obtain",
    "generate_image.description": "an asset the user directed you to retrieve",
    "edit_image.description": "an asset the user directed you to retrieve",
    "generate_video.description": "an asset the user directed you to retrieve",
}


def test_every_built_image_tool_is_classified() -> None:
    """A new tool from the same wrapper needs triage, not silent coverage."""
    built = set(_built_image_descriptions())
    named = {n.split(".")[0] for n in REQUIRED_WORDING} | (
        IMAGE_TOOLS_WITHOUT_A_RETRIEVAL_ROUTE
    )

    assert built <= named, (
        f"unclassified image tools: {sorted(built - named)}. Add each to "
        f"REQUIRED_WORDING with its authorization wording, or to "
        f"IMAGE_TOOLS_WITHOUT_A_RETRIEVAL_ROUTE if it takes no external reference."
    )


def test_every_retrieval_tool_is_classified() -> None:
    """A new web or image tool has to be triaged, not silently uncovered."""
    discovered = set(_discovered_tool_classes())
    classified = (
        set(RETRIEVAL_TOOL_CLASSES)
        | NOT_A_RETRIEVAL_ROUTE
        | GENERIC_REMOTE_ROUTE_OUT_OF_SCOPE
    )

    assert discovered <= classified, (
        f"unclassified remote-capable tools: {sorted(discovered - classified)}. "
        f"Add each to RETRIEVAL_TOOL_CLASSES with its authorization wording, to "
        f"NOT_A_RETRIEVAL_ROUTE if it cannot produce a reusable asset reference, "
        f"or to GENERIC_REMOTE_ROUTE_OUT_OF_SCOPE with a reason."
    )
    assert set(RETRIEVAL_TOOL_CLASSES) <= discovered, (
        f"renamed or removed: {sorted(set(RETRIEVAL_TOOL_CLASSES) - discovered)}"
    )
    # Dead entries hide the fact that a carve-out stopped applying.
    assert classified <= discovered, (
        f"classified but no longer discovered: {sorted(classified - discovered)}"
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


@pytest.mark.parametrize("name", sorted(REQUIRED_WORDING))
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
    """The obtain intent qualifies both routes, not only the surfaced one."""
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
    """A non-empty asset_query filters, so page-wide inspection leaves it empty."""
    fetch = _surfaces()["fetch_web_content.description"]
    assert "leaving asset_query empty so none of the static references" in fetch
    assert "the result limit still applies" in fetch
    query_field = _field(FetchWebContentArgs, "asset_query")
    assert "Leave it empty to list every supported static reference" in query_field
    # The tool parses the initial HTML and caps the result, so promising
    # "everything the page loads" overstates what comes back.
    assert "subject to the tool's result limit" in query_field
    # Lazy-loading attributes ARE extracted (web_content.py:262,276), so the
    # caveat has to name what is really excluded.
    assert "referenced only from CSS or constructed at runtime" in query_field
    assert "lazy-loading attributes such as data-src are" in query_field
    assert "not from the result limit" in query_field


def test_authenticity_is_independent_of_authorization() -> None:
    """A requested retrieval can still return a counterfeit."""
    surfaces = _surfaces()

    assert "however the URL was reached" in surfaces["download_web_asset.description"]
    assert (
        "only when the user supplied its bytes or confirmed the source"
        in (surfaces["download_web_asset.description"])
    )
    assert "unverified until the user confirms it" in surfaces["download_web_asset.url"]


def test_asking_the_user_stops_once_they_have_chosen() -> None:
    """An unscoped "ask the user" reopens a question the user already answered."""
    for name in ("generate_image.description", "edit_image.description"):
        text = _surfaces()[name]
        assert "ask the user how to proceed" in text, name
        assert "act on that choice instead of asking again" in text, name


# One canonical clause, word-for-word on every surface, so agreement is a
# comparison rather than per-side spot checks.
ROUTE_CLAUSE = (
    "URL is only a retrieval route: asking you to retrieve one authorizes the "
    "fetch, not the authenticity, and what comes back stays unverified identity "
    "material until the user confirms its source"
)
CONFIRMATION_CLAUSE = (
    'A user naming a URL as the asset ("here is our logo: <url>") is that '
    "confirmation; a URL that merely served as a fetch route is not."
)


def test_skill_and_tool_descriptions_agree_on_retrieval() -> None:
    """Both land in the same context, so both carry the same canonical clauses."""
    body = " ".join(SkillParser.parse(SKILL_DIR)["content"].split())
    surfaces = _surfaces()

    assert "Two sources need no permission" in body
    assert "official web presence" not in body, (
        "the skill sanctions retrieval the tool descriptions forbid"
    )
    assert "take it only when they tell you to" in body

    for clause in (ROUTE_CLAUSE, CONFIRMATION_CLAUSE):
        assert clause in body, f"skill body dropped: {clause[:50]!r}"
        for name in ("generate_image.description", "edit_image.description"):
            assert clause in surfaces[name], f"{name} dropped: {clause[:50]!r}"

    for name in ("generate_image.description", "edit_image.description"):
        assert "an asset the user directed you to retrieve" in surfaces[name], name


def test_skill_separates_looking_from_taking() -> None:
    """The skill authorizes an external look; that is not authorization to take."""
    body = " ".join(SkillParser.parse(SKILL_DIR)["content"].split())

    assert "Being told to look at a page is not being told to take what it holds" in (
        body
    )
    assert "ask before acquiring or using it" in body


def _system_context_after_load_skill() -> str:
    """The system message an LLM call actually receives once the skill is loaded."""
    import asyncio

    from xagent.core.agent.context.execution import ExecutionContext
    from xagent.core.agent.context.skill_tool import build_load_skill_tool

    skill = SkillParser.parse(SKILL_DIR)
    manager = FakeSkillManager([skill])

    async def run() -> str:
        context = ExecutionContext(system_prompt="Base prompt.")
        tool = await build_load_skill_tool(skill_manager=manager, context=context)
        assert tool is not None
        await tool.execute(skill["name"])
        return str(context.get_messages_for_llm()[0]["content"])

    return " ".join(asyncio.run(run()).split())


def _section(system: str, heading: str, next_heading: str) -> str:
    """One `## ` section of the loaded skill, as it sits in the system message."""
    assert heading in system, f"section missing from loaded context: {heading}"
    body = system.split(heading, 1)[1]
    return body.split(next_heading, 1)[0] if next_heading in body else body


def test_loaded_context_carries_the_policy_without_contradicting_it() -> None:
    """Reading the file proves the text exists; this proves it reaches the model."""
    system = _system_context_after_load_skill()

    assert "Two sources need no permission" in system
    assert "take it only when they tell you to" in system
    assert "URL is only a retrieval route" in system
    # The wording this PR exists to remove must not be reachable from the
    # assembled context either.
    for phrase in ("official web presence", "retrieved with the web tools"):
        assert phrase not in system, f"assembled context still sanctions: {phrase!r}"


def test_each_no_logo_gate_settles_the_choice_in_its_own_section() -> None:
    """Three gates ask for a logo; a global substring passes on any one of them."""
    system = _system_context_after_load_skill()

    source = _section(
        system,
        "## Use brand and reference assets intentionally",
        "## Generate the complete creative",
    )
    assert "act on that choice instead of asking again" in source
    assert "only a later change in what they want reopens the question" in source
    # The identity-asset gate lives in the same section, after the source rules.
    assert "ask how to proceed once" in source
    assert "without asking again" in source
    assert source.index("act on that choice") < source.index("ask how to proceed once")

    gate = _section(system, "## Apply the completion gate", "## Deliver")
    assert "the user has not yet chosen" in gate
    assert "do not reopen the question" in gate
    assert "that draft is the deliverable for this turn" in gate


def test_provenance_reads_the_same_on_every_surface_that_states_it() -> None:
    """Keep the two senses of "supplied" apart on every surface that states it."""
    surfaces = _surfaces()
    system = _system_context_after_load_skill()

    for name in ("generate_image.description", "edit_image.description"):
        text = surfaces[name]
        assert "trusted input" in text, name
        assert "URL is only a retrieval route" in text, name
        assert "authorizes the fetch, not the authenticity" in text, name

    download = surfaces["download_web_asset.description"]
    assert "supplied its bytes or confirmed the source" in download
    assert "Naming a URL as the asset is that confirmation" in download
    assert "merely served as a fetch route is not" in download

    assert "trusted input" in system
    assert "authorizes the fetch, not the" in system


def test_naming_a_url_as_the_asset_confirms_the_source() -> None:
    """Naming a URL as the asset is authorization and confirmation in one."""
    surfaces = _surfaces()
    system = _system_context_after_load_skill()

    assert CONFIRMATION_CLAUSE in system
    for name in ("generate_image.description", "edit_image.description"):
        assert CONFIRMATION_CLAUSE in surfaces[name], name
    assert (
        "Naming a URL as the asset is that confirmation"
        in (surfaces["download_web_asset.description"])
    )


def test_no_surface_frames_self_directed_search_as_a_pipeline_stage() -> None:
    """ "Do not search in parallel" implies searching is expected; assert it is gone."""
    body = " ".join(SkillParser.parse(SKILL_DIR)["content"].split())
    reference = " ".join(
        (SKILL_DIR / "references" / "static-ad-art-direction.md").read_text().split()
    )

    for text, where in ((body, "SKILL.md"), (reference, "reference")):
        assert "Do not search for identity assets in parallel" not in text, where
        assert "Do not search for the logo in parallel" not in text, where

    assert "never by searching on your own" in body
    assert "never from a search you started yourself" in reference
