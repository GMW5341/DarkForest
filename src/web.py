"""
DarkForest Web Server — FastAPI 기반 웹 API.

Multi-Agent Debate 포트폴리오 분석을 REST API로 제공한다.
Human-in-the-Loop: 라운드별 엔드포인트로 사용자 피드백을 수집한다.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analyzer import PortfolioAnalyzer
from src.agents.debate import DebateOrchestrator
from src.models.analysis import (
    AgentOpinion,
    DebatePhase,
    DebateResult,
    DebateRound,
    PortfolioAnalysis,
    UserFeedback,
)
from src.models.portfolio import Holding, Portfolio


# ── In-memory stores ──

class AnalysisJob(BaseModel):
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    result: PortfolioAnalysis | None = None
    error: str | None = None


class DebateSession(BaseModel):
    """인터랙티브 토론 세션 상태."""
    session_id: str
    holding: Holding
    portfolio: Portfolio
    phase: DebatePhase = DebatePhase.WAITING_START
    rounds: list[DebateRound] = Field(default_factory=list)
    opinions: list[AgentOpinion] = Field(default_factory=list)
    user_feedbacks: list[UserFeedback] = Field(default_factory=list)
    result: DebateResult | None = None
    error: str | None = None
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"


_jobs: dict[str, AnalysisJob] = {}
_sessions: dict[str, DebateSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="DarkForest",
    description="Multi-Agent Debate 투자 포트폴리오 분석기 (Human-in-the-Loop)",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 기존 자동 분석 API (호환 유지) ──

class AnalyzeRequest(BaseModel):
    portfolio: Portfolio
    api_key: str | None = Field(default=None, description="Anthropic API 키 (선택)")
    model: str = Field(default="claude-sonnet-4-20250514")
    debate_only: bool = Field(default=False)


class AnalyzeResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: PortfolioAnalysis | None = None
    error: str | None = None


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def start_analysis(req: AnalyzeRequest):
    """포트폴리오 분석 (자동 모드)."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY가 필요합니다.")

    job_id = str(uuid.uuid4())[:8]
    job = AnalysisJob(job_id=job_id, status="running")
    _jobs[job_id] = job

    import asyncio
    asyncio.create_task(
        _run_analysis(job, req.portfolio, api_key, req.model, req.debate_only)
    )
    return AnalyzeResponse(job_id=job_id, status="running")


async def _run_analysis(
    job: AnalysisJob, portfolio: Portfolio,
    api_key: str, model: str, debate_only: bool,
) -> None:
    try:
        analyzer = PortfolioAnalyzer(api_key=api_key, model=model)
        if debate_only:
            result = await analyzer.debate(portfolio)
        else:
            result = await analyzer.analyze(portfolio)
        job.result = result
        job.status = "completed"
    except Exception as e:
        job.error = str(e)
        job.status = "failed"


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return JobStatusResponse(
        job_id=job.job_id, status=job.status,
        result=job.result, error=job.error,
    )


# ── 인터랙티브 토론 API (Human-in-the-Loop) ──

class DebateStartRequest(BaseModel):
    holding: Holding
    portfolio: Portfolio
    api_key: str | None = Field(default=None)
    model: str = Field(default="claude-sonnet-4-20250514")


class FeedbackRequest(BaseModel):
    content: str = Field(default="", description="사용자 코멘트")
    focus_on: list[str] = Field(default_factory=list)
    additional_context: str = Field(default="")
    override_stance: str | None = Field(default=None)


class DebateSessionResponse(BaseModel):
    session_id: str
    phase: DebatePhase
    rounds: list[DebateRound] = Field(default_factory=list)
    result: DebateResult | None = None
    error: str | None = None


def _get_orchestrator(session: DebateSession) -> DebateOrchestrator:
    analyzer = PortfolioAnalyzer(api_key=session.api_key, model=session.model)
    return analyzer.engine.debate_orchestrator


@app.post("/api/debate/start", response_model=DebateSessionResponse)
async def debate_start(req: DebateStartRequest):
    """Round 1 실행: 각 에이전트 독립 분석 → 결과 반환 → 사용자 피드백 대기."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY가 필요합니다.")

    session_id = str(uuid.uuid4())[:8]
    session = DebateSession(
        session_id=session_id,
        holding=req.holding,
        portfolio=req.portfolio,
        api_key=api_key,
        model=req.model,
    )

    try:
        orchestrator = _get_orchestrator(session)
        round1, opinions = await orchestrator.run_round1(
            session.holding, session.portfolio,
        )
        session.rounds.append(round1)
        session.opinions = opinions
        session.phase = DebatePhase.ROUND1_DONE
    except Exception as e:
        session.error = str(e)

    _sessions[session_id] = session
    return DebateSessionResponse(
        session_id=session_id,
        phase=session.phase,
        rounds=session.rounds,
        error=session.error,
    )


@app.post("/api/debate/{session_id}/round2", response_model=DebateSessionResponse)
async def debate_round2(session_id: str, feedback: FeedbackRequest | None = None):
    """사용자 피드백 반영 + Round 2 실행."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if session.phase != DebatePhase.ROUND1_DONE:
        raise HTTPException(status_code=400, detail=f"현재 단계: {session.phase.value}")

    user_fb = None
    if feedback and feedback.content.strip():
        user_fb = UserFeedback(
            content=feedback.content,
            focus_on=feedback.focus_on,
            additional_context=feedback.additional_context,
            override_stance=feedback.override_stance,
        )
        session.user_feedbacks.append(user_fb)

    try:
        orchestrator = _get_orchestrator(session)
        round2 = await orchestrator.run_round2(
            session.holding, session.portfolio,
            session.opinions, user_feedback=user_fb,
        )
        session.rounds.append(round2)
        session.phase = DebatePhase.ROUND2_DONE
    except Exception as e:
        session.error = str(e)

    return DebateSessionResponse(
        session_id=session_id,
        phase=session.phase,
        rounds=session.rounds,
        error=session.error,
    )


@app.post("/api/debate/{session_id}/round3", response_model=DebateSessionResponse)
async def debate_round3(session_id: str, feedback: FeedbackRequest | None = None):
    """사용자 피드백 반영 + Round 3 실행."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if session.phase != DebatePhase.ROUND2_DONE:
        raise HTTPException(status_code=400, detail=f"현재 단계: {session.phase.value}")

    user_fb = None
    if feedback and feedback.content.strip():
        user_fb = UserFeedback(
            content=feedback.content,
            focus_on=feedback.focus_on,
            additional_context=feedback.additional_context,
            override_stance=feedback.override_stance,
        )
        session.user_feedbacks.append(user_fb)

    try:
        orchestrator = _get_orchestrator(session)
        round3 = await orchestrator.run_round3(
            session.holding, session.portfolio,
            session.rounds, user_feedback=user_fb,
        )
        session.rounds.append(round3)
        session.phase = DebatePhase.ROUND3_DONE
    except Exception as e:
        session.error = str(e)

    return DebateSessionResponse(
        session_id=session_id,
        phase=session.phase,
        rounds=session.rounds,
        error=session.error,
    )


@app.post("/api/debate/{session_id}/synthesize", response_model=DebateSessionResponse)
async def debate_synthesize(session_id: str, feedback: FeedbackRequest | None = None):
    """사용자 최종 피드백 반영 + 최종 판정."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if session.phase != DebatePhase.ROUND3_DONE:
        raise HTTPException(status_code=400, detail=f"현재 단계: {session.phase.value}")

    final_comment = ""
    if feedback and feedback.content.strip():
        user_fb = UserFeedback(
            content=feedback.content,
            focus_on=feedback.focus_on,
            additional_context=feedback.additional_context,
            override_stance=feedback.override_stance,
        )
        session.user_feedbacks.append(user_fb)
        final_comment = feedback.content

    try:
        orchestrator = _get_orchestrator(session)
        synthesis = await orchestrator.synthesize(
            session.holding, session.rounds,
            session.user_feedbacks, final_comment,
        )

        session.result = DebateResult(
            target_company=session.holding.name,
            target_ticker=session.holding.ticker,
            rounds=session.rounds,
            consensus_points=synthesis.get("consensus_points", []),
            dissent_points=synthesis.get("dissent_points", []),
            final_decision=orchestrator._parse_decision(
                synthesis.get("final_decision", "needs_more_data")
            ),
            final_confidence=min(
                max(synthesis.get("final_confidence", 0.5), 0.0), 1.0
            ),
            final_reasoning=synthesis.get("final_reasoning", ""),
            action_items=synthesis.get("action_items", []),
            risk_summary=synthesis.get("risk_summary", ""),
            price_assessment=synthesis.get("price_assessment", ""),
            user_feedbacks=session.user_feedbacks,
            phase=DebatePhase.SYNTHESIZED,
        )
        session.phase = DebatePhase.SYNTHESIZED
    except Exception as e:
        session.error = str(e)

    return DebateSessionResponse(
        session_id=session_id,
        phase=session.phase,
        rounds=session.rounds,
        result=session.result,
        error=session.error,
    )


@app.get("/api/debate/{session_id}", response_model=DebateSessionResponse)
async def get_debate_session(session_id: str):
    """토론 세션 상태 조회."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return DebateSessionResponse(
        session_id=session_id,
        phase=session.phase,
        rounds=session.rounds,
        result=session.result,
        error=session.error,
    )


# ── 기타 ──

@app.post("/api/analyze/sync", response_model=PortfolioAnalysis)
async def analyze_sync(req: AnalyzeRequest):
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY가 필요합니다.")
    try:
        analyzer = PortfolioAnalyzer(api_key=api_key, model=req.model)
        if req.debate_only:
            return await analyzer.debate(req.portfolio)
        return await analyzer.analyze(req.portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "darkforest", "version": "0.3.0"}


# ── Static files ──
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "DarkForest API is running. POST /api/debate/start to begin."}


if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
