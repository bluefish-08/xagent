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
    assert "complete PNG or JPEG assets" in description
    assert "commercial and brand-facing" in description
    assert "campaign posters" in description
    assert "advertising creatives" in description
    assert "placement variants" in description
    assert "marketing, campaign, event, or brand communication" in when_to_use
    assert "educational infographics" in when_to_use
    assert "technical diagrams" in when_to_use
    assert "concept explainers" in when_to_use

    # Auto routing sees bounded one-line versions of these fields. Keep the
    # positive commercial scope and the important exclusions inside that
    # actual routing surface instead of only in the full skill body.
    routing_description = _index_text(skill["description"])
    routing_when_to_use = _index_text(skill["when_to_use"])
    assert "commercial and brand-facing" in routing_description
    assert "advertising creatives" in routing_description
    assert "Use only for marketing" in routing_when_to_use
    assert "educational infographics" in routing_when_to_use
    assert "concept explainers" in routing_when_to_use

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
    assert "The user's other tasks and earlier outputs are not a source" in content
    assert "recover by finding it with `list_all_user_files`" in content
    # Three definitions the rules below reference instead of restating. Restating
    # them is how eight sites drifted into contradicting each other.
    assert "A **required asset** is one deliverable" in content
    assert "Coverage is never" in content
    assert "budgeted; it is what the deliverable is" in content
    assert "any `generate_image` or `edit_image` call on an asset that" in content
    assert "at most two on the same asset, and" in content
    assert "four across the run" in content
    # Optional assets cost from their first call, or self-invented directions are
    # neither coverage nor repair and nothing bounds them.
    assert "An optional asset costs a repair for every call it takes" in content
    # Two gates. Coverage never yields to the budget; quality does.
    assert "**Coverage is unconditional.**" in content
    assert "spent budget is never a reason an asset is missing" in content
    assert "**Quality is what the budget releases.**" in content
    # Coverage failures must not sit in the budget-conditional list.
    assert "Some failures are coverage failures" in content
    assert "These are not budgeted" in content
    # The hand-back has to survive the completion check, and the two fields live
    # on two different tools.
    assert "leaving the decision's missing-verification field empty" in content
    assert "outcome `partial`" in content
    assert "it gives up after a few" in content
    # A handed-down stop rule cannot be edited, so the step reports it unmet.
    assert "report the condition as unmet" in content
    assert "let the user decide whether to spend another round" in content


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
    # Mandatory reading that lands in the same context as the skill, so its
    # rejection rules need the budget too.
    assert "within the limits the main skill sets on repairs" in content
    # Coverage failures here must not inherit the budget the quality ones do.
    assert "A failure that leaves a required asset missing" in content
    assert "not budgeted and must be fixed" in content
    assert (
        "subordinate to the main\nskill's repair budget" in reference_path.read_text()
    )
