from pathlib import Path

from xagent.core.agent.context.skill_tool import _index_text
from xagent.skills.parser import SkillParser


def test_static_visual_design_skill_routes_only_commercial_creatives() -> None:
    skill_dir = (
        Path(__file__).parents[2]
        / "src"
        / "xagent"
        / "skills"
        / "builtin"
        / "static-visual-design"
    )

    skill = SkillParser.parse(skill_dir)

    assert skill["name"] == "static-visual-design"
    description = " ".join(skill["description"].split())
    when_to_use = " ".join(skill["when_to_use"].split())
    assert "one PNG/JPEG" in description
    assert "brand styling" in description
    assert "finished ad, poster, banner, social post" in description
    assert "another placement size or aspect ratio" in description
    assert "marketing, promotion, campaign, event, or brand-facing image" in when_to_use
    assert "explanatory diagrams" in when_to_use
    assert "infographics" in when_to_use
    assert "plain illustrations" in when_to_use

    # Auto routing sees bounded one-line versions of these fields, so every
    # assertion above has to hold on the routing surface, not only in the body.
    assert _index_text(skill["description"]) == description
    assert _index_text(skill["when_to_use"]) == when_to_use

    content = " ".join(skill["content"].split())
    assert "Stay within the commercial-creative scope" in content
    assert "Use `generate_image` to create the full designed asset" in content
    assert "references/static-ad-art-direction.md" in content
    assert "two or three genuinely different communication angles" in content
    assert "one finished placement on one continuous canvas" in content
    assert "a brand-specific final requires a verified logo" in content
    assert "This runtime does not provide deterministic compositing" in content
    assert "download_web_asset" not in content
    assert "SVG is source text" not in content
    assert "Do not use HTML/CSS plus browser screenshots" in content
    assert "Do not enter `final_answer`" in content
    assert "Return only final PNG or JPEG files" in content


def test_static_visual_design_includes_art_direction_reference() -> None:
    reference_path = (
        Path(__file__).parents[2]
        / "src"
        / "xagent"
        / "skills"
        / "builtin"
        / "static-visual-design"
        / "references"
        / "static-ad-art-direction.md"
    )

    content = " ".join(reference_path.read_text().split())

    assert "Choose a communication structure" in content
    assert "Dominant proof" in content
    assert "Design for a three-pass read" in content
    assert "Follow the main skill's one-canvas generation contract" in content
    assert "Automatic rejection overrides subjective scoring" in content
