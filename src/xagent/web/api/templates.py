"""
Templates API Endpoints

Provides REST API endpoints for managing and using agent templates.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.template_stats import TemplateStats
from ..models.user import User

logger = logging.getLogger(__name__)


# ===== Helper Functions =====


def get_localized_description(
    descriptions: dict[str, str], lang: Optional[str] = None
) -> str:
    """
    根据语言偏好获取本地化的描述

    Args:
        descriptions: 描述字典 {en: "...", zh: "..."}
        lang: 语言代码，如果为 None 则尝试从英文 fallback

    Returns:
        本地化的描述字符串
    """
    if not descriptions:
        return ""

    # 如果指定了语言且存在，返回该语言
    if lang and lang in descriptions:
        return descriptions[lang]

    # Fallback to English
    return descriptions.get("en", "")


# ===== Pydantic Models =====


class AgentConfig(BaseModel):
    """Agent configuration from template"""

    instructions: str = Field(..., description="System prompt/instructions")
    skills: list[str] = Field(default_factory=list, description="List of skill names")
    tool_categories: list[str] = Field(
        default_factory=list, description="List of tool categories"
    )
    execution_mode: str = Field(
        default="react", description="Execution mode: simple, react, or graph"
    )


class TemplateInfo(BaseModel):
    """Template brief information"""

    id: str = Field(..., description="Template unique identifier")
    name: str = Field(..., description="Template name")
    category: str = Field(..., description="Template category")
    featured: bool = Field(
        default=False, description="Whether the template is featured"
    )
    description: str = Field(..., description="Template description")
    tags: list[str] = Field(default_factory=list, description="Template tags")
    author: str = Field(..., description="Template author")
    version: str = Field(..., description="Template version")
    views: int = Field(default=0, description="Number of views")
    likes: int = Field(default=0, description="Number of likes")
    used_count: int = Field(default=0, description="Number of times used")


class TemplateDetail(TemplateInfo):
    """Template complete information with agent config"""

    agent_config: AgentConfig = Field(..., description="Agent configuration")


class LikeResponse(BaseModel):
    """Like/unlike response"""

    liked: bool = Field(..., description="Whether the template is liked")
    likes: int = Field(..., description="Total number of likes")


# ===== Router =====

router = APIRouter(prefix="/api/templates", tags=["templates"])


# ===== Helper Functions =====


def get_or_create_template_stats(db: Session, template_id: str) -> TemplateStats:
    """Get or create template stats record"""
    stats = (
        db.query(TemplateStats).filter(TemplateStats.template_id == template_id).first()
    )
    if not stats:
        stats = TemplateStats(template_id=template_id)
        db.add(stats)
        db.commit()
        db.refresh(stats)
    return stats


# ===== Endpoints =====


@router.get("/", response_model=list[TemplateInfo])
async def list_templates(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lang: Optional[str] = Query(None, description="Language code (e.g., 'en', 'zh')"),
) -> list[TemplateInfo]:
    """
    列出所有可用的 templates（包含统计数据）

    Args:
        lang: Optional language code for localized descriptions

    Returns:
        List of available templates with statistics
    """
    template_manager = request.app.state.template_manager
    templates = await template_manager.list_templates()

    # Get statistics from database
    result = []
    for template in templates:
        template_id = template["id"]
        stats = get_or_create_template_stats(db, template_id)

        # Get localized description
        descriptions = template.get("descriptions", {})
        description = get_localized_description(descriptions, lang)

        result.append(
            TemplateInfo(
                id=template["id"],
                name=template["name"],
                category=template.get("category", ""),
                featured=bool(template.get("featured", False)),
                description=description,
                tags=template.get("tags", []),
                author=template.get("author", ""),
                version=template.get("version", ""),
                views=stats.views,
                likes=stats.likes,
                used_count=stats.used_count,
            )
        )

    return result


@router.get("/{template_id}", response_model=TemplateDetail)
async def get_template(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lang: Optional[str] = Query(None, description="Language code (e.g., 'en', 'zh')"),
) -> TemplateDetail:
    """
    获取单个 template 详情（包含 agent_config）

    Args:
        template_id: ID of the template to retrieve
        lang: Optional language code for localized descriptions

    Returns:
        Detailed template information with agent configuration

    Raises:
        HTTPException: If template not found
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Get statistics from database
    stats = get_or_create_template_stats(db, template_id)

    # Increment view count
    stats.views += 1
    db.commit()

    # Get localized description
    descriptions = template.get("descriptions", {})
    description = get_localized_description(descriptions, lang)

    return TemplateDetail(
        id=template["id"],
        name=template["name"],
        category=template.get("category", ""),
        featured=bool(template.get("featured", False)),
        description=description,
        tags=template.get("tags", []),
        author=template.get("author", ""),
        version=template.get("version", ""),
        views=stats.views,
        likes=stats.likes,
        used_count=stats.used_count,
        agent_config=AgentConfig(
            instructions=template["agent_config"].get("instructions", ""),
            skills=template["agent_config"].get("skills", []),
            tool_categories=template["agent_config"].get("tool_categories", []),
        ),
    )


@router.post("/{template_id}/like", response_model=LikeResponse)
async def like_template(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LikeResponse:
    """
    点赞或取消点赞 template

    Args:
        template_id: ID of the template to like/unlike

    Returns:
        Current like status and total likes

    Raises:
        HTTPException: If template not found
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    stats = get_or_create_template_stats(db, template_id)

    # Simple toggle like (in production, track user-specific likes)
    stats.likes += 1
    db.commit()

    return LikeResponse(liked=True, likes=stats.likes)


@router.post("/{template_id}/use")
async def use_template(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    使用 template 创建 agent（记录使用次数）

    Args:
        template_id: ID of the template to use

    Returns:
        Success message

    Raises:
        HTTPException: If template not found
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Increment used count
    stats = get_or_create_template_stats(db, template_id)
    stats.used_count += 1
    db.commit()

    return {
        "message": "Template usage recorded",
        "template_id": template_id,
        "used_count": stats.used_count,
    }
