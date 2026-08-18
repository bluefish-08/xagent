"""Asset-retrieval tool descriptions must not authorize unprompted retrieval.

These descriptions are injected into the tool schema on every tool-capable step,
while a skill's policy is only present once it has been loaded. A description
that offers a wider route than the skill allows therefore wins by default, so the
authorization wording has to live here too.
"""

import pytest

from xagent.core.tools.adapters.vibe.download_web_asset import (
    DownloadWebAssetArgs,
    DownloadWebAssetTool,
)
from xagent.core.tools.adapters.vibe.fetch_web_content import (
    FetchWebContentArgs,
    FetchWebContentTool,
)
from xagent.core.tools.core.image_tool import ImageGenerationToolCore


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _field(model: type, name: str) -> str:
    return _normalized(model.model_fields[name].description or "")


def _surfaces() -> dict[str, str]:
    """Every description reaching the model alongside an asset-fetch route."""
    return {
        "fetch_web_content.description": _normalized(FetchWebContentTool().description),
        "fetch_web_content.include_assets": _field(
            FetchWebContentArgs, "include_assets"
        ),
        "download_web_asset.description": _normalized(
            DownloadWebAssetTool(None).description
        ),
        "download_web_asset.url": _field(DownloadWebAssetArgs, "url"),
        "generate_image.description": _normalized(
            ImageGenerationToolCore.GENERATE_IMAGE_DESCRIPTION
        ),
        "edit_image.description": _normalized(
            ImageGenerationToolCore.EDIT_IMAGE_DESCRIPTION
        ),
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
    "fetch_web_content.description": "When the user asked you to obtain an exact asset",
    "fetch_web_content.include_assets": (
        "Enable it when the user asked you to inspect or enumerate what a page loads"
    ),
    "download_web_asset.description": (
        "The user has to have asked you to obtain this asset"
    ),
    "download_web_asset.url": "one the user supplied directly",
    "generate_image.description": "an asset the user directed you to retrieve",
    "edit_image.description": "an asset the user directed you to retrieve",
}


@pytest.mark.parametrize("name", sorted(_surfaces()))
def test_surface_forbids_unprompted_retrieval(name: str) -> None:
    lowered = _surfaces()[name].lower()
    for phrase in BANNED_WORDING:
        assert phrase not in lowered, f"{name} reintroduced: {phrase!r}"


def test_every_surface_is_covered() -> None:
    assert set(_surfaces()) == set(REQUIRED_WORDING), (
        "a retrieval surface was added or renamed without its authorization wording"
    )


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


def test_direct_user_urls_are_a_described_download_route() -> None:
    """The user pasting an exact URL is already authorized; describe it as such."""
    for name in ("download_web_asset.description", "download_web_asset.url"):
        assert "supplied directly" in _surfaces()[name], name


def test_enumeration_does_not_authorize_a_download() -> None:
    """include_assets is inspect-only, so a URL it lists is not yet acquirable.

    Without this boundary the widened enumeration wording would let an
    inspect-only request satisfy download_web_asset's authorization clause.
    """
    surfaces = _surfaces()

    assert (
        "does not authorize download_web_asset"
        in (surfaces["fetch_web_content.include_assets"])
    )
    assert "does not authorize a download" in surfaces["download_web_asset.description"]
    assert "is not such a request" in surfaces["download_web_asset.url"]


def test_authenticity_is_independent_of_authorization() -> None:
    """A requested retrieval can still return a counterfeit."""
    surfaces = _surfaces()

    assert "however the URL was reached" in surfaces["download_web_asset.description"]
    assert (
        "only when the user supplied it or confirmed the source"
        in (surfaces["download_web_asset.description"])
    )
    assert "unverified until the user confirms it" in surfaces["download_web_asset.url"]


def test_include_assets_is_also_an_enumeration_contract() -> None:
    """The flag lists page resources without downloading them.

    Scoping it to obtaining an asset would block legitimate requests to inspect a
    page's icons, manifest, scripts, or broken asset references.
    """
    flag = _surfaces()["fetch_web_content.include_assets"]

    assert "inspect or enumerate" in flag
    assert "nothing is downloaded" in flag
    assert "asset-hunting on your own" in flag


def test_asking_the_user_stops_once_they_have_chosen() -> None:
    """An unscoped "ask the user" reopens a question the user already answered."""
    for name in ("generate_image.description", "edit_image.description"):
        text = _surfaces()[name]
        assert "ask the user how to proceed" in text, name
        assert "act on that choice instead of asking again" in text, name
