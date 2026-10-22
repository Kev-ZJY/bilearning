"""ORM 实体定义：Project / Video / Message / KnowledgeDoc。

关系模型（核心产品概念）：
- Project  ↔ KnowledgeDoc : 1:1（project_id 唯一）
- Project  ↔ Video        : 1:N
- Project  ↔ Message      : 1:N

AppSetting 是**全局**表（没有 project_id），不属于上面的关系模型：
LLM 配置是整个后端一份，不是每个项目一份。

KnowledgeDoc.doc_json 是产品核心资产：LLM 产出结构化 JSON（不直接写 HTML），
由右侧面板做折叠/搜索/锚点/掌握度勾选等交互，并支持增量更新与多格式导出。
所有 *_json 字段以 TEXT 存储 JSON 字符串，避免引入数据库特定 JSON 类型。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    # Python 3.13 中 datetime.utcnow 已废弃，统一使用带时区的 now(timezone.utc)
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    subject: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    storage_root: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    videos: Mapped[list["Video"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    doc: Mapped[Optional["KnowledgeDoc"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", uselist=False
    )


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    bvid: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cid: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="bilibili", nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(nullable=True)
    cover: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    artifacts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extracted_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="videos")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_refs_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    project: Mapped["Project"] = relationship(back_populates="messages")


class ConversationSummary(Base):
    """项目级、带源消息校验的会话历史摘要。

    Message 原文永远保留。该表只保存模型上下文用的摘要表示、CAS 版本、
    被纳入的消息 ID/内容哈希，以及最后一个完整用户轮次游标。
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_hash: Mapped[str] = mapped_column(String, nullable=False)
    source_message_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_message_hashes_json: Mapped[str] = mapped_column(Text, nullable=False)
    covered_until_user_message_id: Mapped[Optional[int]] = mapped_column(nullable=True)
    snapshot_anchor_message_id: Mapped[int] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class KnowledgeDoc(Base):
    __tablename__ = "knowledge_docs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    doc_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )

    project: Mapped["Project"] = relationship(back_populates="doc")


class AppSetting(Base):
    """全局键值配置（目前只有 LLM 一组）。

    为什么是键值表而不是给 LLM 建一张带列的表
    ----------------------------------------
    落库的字段会随功能增长（先 LLM base_url/model/max_tokens/timeout，
    以后可能有 ASR 参数、导出选项）。键值表加一个配置项只要插一行，
    而带列的表要改 schema + 写迁移。同时键值表天然支持「只覆盖用户改过的
    那一项」——这是下面 :func:`app.core.config.get_settings` 的核心语义。

    为什么要有这张表（用户报「你的后端 LLM 配置入口缺失」）
    ------------------------------------------------------
    改一次 LLM 配置就要重启后端、重新 source .env。对一个本地工具来说
    这不是「不方便」，是**不可用**：换模型试一下要重启一次。

    优先级：数据库 > 环境变量。
    环境变量不是被取代，而是**首次启动时的默认值** —— 部署时用 .env 铺好，
    之后在界面上随手改。
    """

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow
    )


class KnowledgeRevision(Base):
    """替换前的完整文档；新增独立表，不更改既有表结构。"""
    __tablename__ = "knowledge_revisions"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False)
    doc_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
