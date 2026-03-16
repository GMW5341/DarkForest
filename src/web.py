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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.analyzer import PortfolioAnalyzer
from src.agents.debate import DebateOrchestrator
from src.agents.macro_base import MacroAgentOpinion
from src.agents.macro_debate import MacroDebateOrchestrator, MacroDebateRound
from src.agents.macro_economist import MacroEconomistAgent
from src.agents.market_strategist import MarketStrategistAgent
from src.agents.systemic_risk import SystemicRiskAgent
from src.api.client import ClaudeClient
from src.models.analysis import (
    AgentOpinion,
    DebatePhase,
    DebateResult,
    DebateRound,
    PortfolioAnalysis,
    UserFeedback,
)
from src.models.macro import (
    MacroDebateResult,
    MacroDebateRoundModel,
    MacroMessageModel,
    MacroTopic,
)
from src.agents.macro_base import MacroDebateMessage
from src.config import UsageTracker, get_settings, update_settings
from src.data.history import get_document_store, get_draft_store, get_history_store
from src.data.filings import fetch_filing_data
from src.data.market_data import fetch_macro_market_data, fetch_market_snapshot, fetch_stock_financials
from src.data.pdf_extractor import extract_text_from_file, extract_text_from_file_with_vision
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
    model: str = "claude-opus-4-20250514"
    market_context: str = Field(default="", description="실시간 시장 데이터 컨텍스트")


_jobs: dict[str, AnalysisJob] = {}
_sessions: dict[str, DebateSession] = {}
_macro_sessions: dict[str, "MacroDebateSession"] = {}


# ── 직렬화 헬퍼 (세션 ↔ JSON) ──


def _serialize_stock_session(session: DebateSession) -> dict[str, Any]:
    """Stock DebateSession → JSON dict (드래프트 저장용)."""
    return {
        "mode": "stock",
        "session_id": session.session_id,
        "holding": session.holding.model_dump(),
        "portfolio": session.portfolio.model_dump(),
        "phase": session.phase.value,
        "rounds": [r.model_dump() for r in session.rounds],
        "opinions": [o.model_dump() for o in session.opinions],
        "user_feedbacks": [fb.model_dump() for fb in session.user_feedbacks],
        "api_key": session.api_key,
        "model": session.model,
        "market_context": session.market_context,
        "topic": f"{session.holding.name} ({session.holding.ticker})",
        "round_count": len(session.rounds),
    }


def _deserialize_stock_session(data: dict[str, Any]) -> DebateSession:
    """JSON dict → Stock DebateSession."""
    return DebateSession(
        session_id=data["session_id"],
        holding=Holding(**data["holding"]),
        portfolio=Portfolio(**data["portfolio"]),
        phase=DebatePhase(data["phase"]),
        rounds=[DebateRound(**r) for r in data.get("rounds", [])],
        opinions=[AgentOpinion(**o) for o in data.get("opinions", [])],
        user_feedbacks=[UserFeedback(**fb) for fb in data.get("user_feedbacks", [])],
        api_key=data.get("api_key", ""),
        model=data.get("model", "claude-sonnet-4-20250514"),
        market_context=data.get("market_context", ""),
    )


def _serialize_macro_session(session: MacroDebateSession) -> dict[str, Any]:
    """Macro MacroDebateSession → JSON dict (드래프트 저장용)."""
    rounds_data = []
    for r in session.rounds:
        rounds_data.append({
            "round_number": r.round_number,
            "round_type": r.round_type,
            "messages": [
                {
                    "agent_name": m.agent_name,
                    "round_number": m.round_number,
                    "message_type": m.message_type,
                    "stance": m.stance,
                    "confidence": m.confidence,
                    "content": m.content,
                    "agreements": m.agreements,
                    "disagreements": m.disagreements,
                }
                for m in r.messages
            ],
        })

    opinions_data = []
    for o in session.opinions:
        opinions_data.append({
            "agent_name": o.agent_name,
            "agent_role": o.agent_role,
            "stance": o.stance,
            "confidence": o.confidence,
            "reasoning": o.reasoning,
            "key_points": o.key_points,
            "risks": o.risks,
            "action_items": o.action_items,
        })

    return {
        "mode": "macro",
        "session_id": session.session_id,
        "topic": session.topic,
        "context": session.context,
        "api_key": session.api_key,
        "model": session.model,
        "market_context": session.market_context,
        "phase": session.phase,
        "rounds": rounds_data,
        "opinions": opinions_data,
        "user_feedbacks": [fb.model_dump() for fb in session.user_feedbacks],
        "round_count": len(session.rounds),
    }


def _deserialize_macro_session(data: dict[str, Any]) -> MacroDebateSession:
    """JSON dict → Macro MacroDebateSession."""
    session = MacroDebateSession(
        session_id=data["session_id"],
        topic=data["topic"],
        context=data.get("context", ""),
        api_key=data.get("api_key", ""),
        model=data.get("model", "claude-sonnet-4-20250514"),
        market_context=data.get("market_context", ""),
    )
    session.phase = data.get("phase", "waiting_start")

    for r_data in data.get("rounds", []):
        messages = [
            MacroDebateMessage(**m) for m in r_data.get("messages", [])
        ]
        session.rounds.append(MacroDebateRound(
            round_number=r_data["round_number"],
            round_type=r_data["round_type"],
            messages=messages,
        ))

    for o_data in data.get("opinions", []):
        session.opinions.append(MacroAgentOpinion(**o_data))

    for fb_data in data.get("user_feedbacks", []):
        session.user_feedbacks.append(UserFeedback(**fb_data))

    return session


def _save_stock_draft(session: DebateSession) -> None:
    """Stock 세션 중간 저장."""
    try:
        snapshot = _serialize_stock_session(session)
        get_draft_store().save(session.session_id, snapshot)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Stock draft save failed: {e}")


def _save_macro_draft(session: MacroDebateSession) -> None:
    """Macro 세션 중간 저장."""
    try:
        snapshot = _serialize_macro_session(session)
        get_draft_store().save(session.session_id, snapshot)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Macro draft save failed: {e}")


def _restore_drafts_to_memory() -> None:
    """서버 시작 시 미완료 드래프트를 메모리에 복원."""
    store = get_draft_store()
    for draft_summary in store.list_all(limit=100):
        sid = draft_summary["session_id"]
        data = store.load(sid)
        if not data:
            continue
        try:
            mode = data.get("mode", "")
            if mode == "stock" and sid not in _sessions:
                _sessions[sid] = _deserialize_stock_session(data)
            elif mode == "macro" and sid not in _macro_sessions:
                _macro_sessions[sid] = _deserialize_macro_session(data)
        except Exception:
            continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    _restore_drafts_to_memory()
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
    message: str = Field(default="", description="자연어 피드백 (대화하듯이 입력)")


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

    # 실시간 시장 데이터 수집
    market_context = ""
    try:
        snapshot = await fetch_market_snapshot(
            ticker=req.holding.ticker,
            ticker_name=req.holding.name,
        )
        market_context = snapshot.to_context_text()
    except Exception:
        pass  # 시장 데이터 실패 시 무시

    # 종목 재무 데이터 자동 수집 (ticker만 있으면 자동으로 가져옴)
    financial_context = ""
    try:
        financials = await fetch_stock_financials(
            ticker=req.holding.ticker,
            name=req.holding.name,
        )
        financial_context = financials.to_context_text()
    except Exception:
        pass  # 재무 데이터 실패 시 무시

    # 공시 원문 재무제표 자동 수집 (DART / SEC EDGAR)
    filing_context = ""
    try:
        from src.config import get_settings
        dart_key = get_settings().dart_api_key
        filing = await fetch_filing_data(
            ticker=req.holding.ticker,
            name=req.holding.name,
            dart_api_key=dart_key,
        )
        filing_context = filing.to_context_text()
    except Exception:
        pass  # 공시 데이터 실패 시 무시

    # 시장 데이터 + 재무 데이터 + 공시 데이터 + 누적 문서 + 과거 인사이트를 holding에 추가
    holding = req.holding.model_copy()
    session_id = str(uuid.uuid4())[:8]
    extra_context = ""
    if filing_context:
        extra_context += filing_context
    if financial_context:
        extra_context += financial_context
    if market_context:
        extra_context += market_context
    # 영구 저장된 문서 베이스 주입
    doc_context = get_document_store().get_all_context()
    if doc_context:
        extra_context += doc_context
    # 과거 토론 인사이트 (종목 토론에도 피드백 루프 적용)
    past_insights = get_history_store().get_recent_insights(limit=5)
    if past_insights:
        past_lines = ["\n\n## 과거 토론 인사이트 (참고용)"]
        for h in past_insights:
            past_lines.append(
                f"- [{h['mode'].upper()}] {h['topic']}: {h.get('overall_stance', '?')} "
                f"— {h.get('executive_summary', '')[:100]}"
            )
        extra_context += "\n".join(past_lines)
    if extra_context:
        existing = holding.financial_data or ""
        holding.financial_data = existing + "\n" + extra_context if existing else extra_context
    session = DebateSession(
        session_id=session_id,
        holding=holding,
        portfolio=req.portfolio,
        api_key=api_key,
        model=req.model,
        market_context=market_context,
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
    if not session.error:
        _save_stock_draft(session)
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
    if feedback and feedback.message.strip():
        user_fb = UserFeedback(content=feedback.message.strip())
        session.user_feedbacks.append(user_fb)

    try:
        orchestrator = _get_orchestrator(session)
        round2 = await orchestrator.run_round2(
            session.holding, session.portfolio,
            session.opinions, user_feedback=user_fb,
        )
        session.rounds.append(round2)
        session.phase = DebatePhase.ROUND2_DONE
        _save_stock_draft(session)
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
    if feedback and feedback.message.strip():
        user_fb = UserFeedback(content=feedback.message.strip())
        session.user_feedbacks.append(user_fb)

    try:
        orchestrator = _get_orchestrator(session)
        round3 = await orchestrator.run_round3(
            session.holding, session.portfolio,
            session.rounds, user_feedback=user_fb,
        )
        session.rounds.append(round3)
        session.phase = DebatePhase.ROUND3_DONE
        _save_stock_draft(session)
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
    if feedback and feedback.message.strip():
        user_fb = UserFeedback(content=feedback.message.strip())
        session.user_feedbacks.append(user_fb)
        final_comment = feedback.message.strip()

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
            mcda=synthesis.get("mcda"),
            user_feedbacks=session.user_feedbacks,
            phase=DebatePhase.SYNTHESIZED,
        )
        session.phase = DebatePhase.SYNTHESIZED

        # 토론 이력 저장 (영구 저장)
        history_record = {
            "session_id": session_id,
            "mode": "stock",
            "topic": f"{session.holding.name} ({session.holding.ticker})",
            "overall_stance": synthesis.get("final_decision", ""),
            "executive_summary": synthesis.get("final_reasoning", ""),
            "key_insights": synthesis.get("consensus_points", []),
            "dissent_points": synthesis.get("dissent_points", []),
            "confidence": synthesis.get("final_confidence", 0.5),
            "action_items": synthesis.get("action_items", []),
            "risk_summary": synthesis.get("risk_summary", ""),
            "round_count": len(session.rounds),
            "user_feedbacks": [fb.content for fb in session.user_feedbacks],
            "rounds_summary": [
                {
                    "round_number": r.round_number,
                    "messages": [
                        {"agent": m.agent_name, "stance": m.stance, "content": m.content[:300]}
                        for m in r.messages
                    ],
                }
                for r in session.rounds
            ],
        }
        _debate_history_store.append(history_record)
        get_history_store().save(session_id, history_record)
        # 완료된 토론의 드래프트 삭제
        get_draft_store().delete(session_id)
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


# ── 거시 경제 토론 API ──

class MacroDebateSession:
    """거시 경제 토론 세션 상태."""

    def __init__(
        self,
        session_id: str,
        topic: str,
        context: str,
        api_key: str,
        model: str,
        market_context: str = "",
    ):
        self.session_id = session_id
        self.topic = topic
        self.context = context
        self.api_key = api_key
        self.model = model
        self.market_context = market_context
        self.phase = "waiting_start"
        self.rounds: list[MacroDebateRound] = []
        self.opinions: list[MacroAgentOpinion] = []
        self.user_feedbacks: list[UserFeedback] = []
        self.result: dict | None = None
        self.error: str | None = None
        self.usage_tracker = UsageTracker()


class MacroStartRequest(BaseModel):
    topic: MacroTopic
    api_key: str | None = Field(default=None)
    model: str = Field(default="claude-opus-4-20250514")
    include_past_insights: bool = Field(
        default=True, description="과거 토론 인사이트를 배경 정보에 포함할지 여부"
    )


class MacroSessionResponse(BaseModel):
    session_id: str
    phase: str
    rounds: list[MacroDebateRoundModel] = Field(default_factory=list)
    result: MacroDebateResult | None = None
    error: str | None = None
    usage: dict | None = None


def _get_macro_orchestrator(session: MacroDebateSession) -> MacroDebateOrchestrator:
    client = ClaudeClient(
        api_key=session.api_key,
        model=session.model,
        usage_tracker=session.usage_tracker,
    )
    agents = [
        MacroEconomistAgent(client),
        MarketStrategistAgent(client),
        SystemicRiskAgent(client),
    ]
    return MacroDebateOrchestrator(agents=agents, synthesizer_client=client)


def _round_to_model(rnd: MacroDebateRound) -> MacroDebateRoundModel:
    return MacroDebateRoundModel(
        round_number=rnd.round_number,
        round_type=rnd.round_type,
        messages=[
            MacroMessageModel(
                agent_name=m.agent_name,
                round_number=m.round_number,
                message_type=m.message_type,
                stance=m.stance,
                confidence=m.confidence,
                content=m.content,
                agreements=m.agreements,
                disagreements=m.disagreements,
            )
            for m in rnd.messages
        ],
    )


def _macro_response(session: MacroDebateSession, result: MacroDebateResult | None = None) -> MacroSessionResponse:
    return MacroSessionResponse(
        session_id=session.session_id,
        phase=session.phase,
        rounds=[_round_to_model(r) for r in session.rounds],
        result=result,
        error=session.error,
        usage=session.usage_tracker.summary(),
    )


@app.post("/api/macro/start", response_model=MacroSessionResponse)
async def macro_start(req: MacroStartRequest):
    """거시 경제 토론 Round 1 시작."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY가 필요합니다.")

    # 실시간 시장 데이터 수집
    market_context = ""
    try:
        snapshot = await fetch_macro_market_data(include_commodities=True)
        market_context = snapshot.to_context_text()
    except Exception:
        pass

    # 과거 토론 인사이트 + 누적 문서 주입
    session_id = str(uuid.uuid4())[:8]
    context = req.topic.context
    if market_context:
        context += "\n" + market_context
    # 영구 저장된 문서 베이스 주입
    doc_context = get_document_store().get_all_context()
    if doc_context:
        context += doc_context
    if req.include_past_insights:
        # 영구 저장소에서 과거 인사이트 로드
        past_insights = get_history_store().get_recent_insights(limit=5)
        if past_insights:
            past_lines = ["\n\n## 과거 토론 인사이트 (참고용)"]
            for h in past_insights:
                past_lines.append(
                    f"- [{h['mode'].upper()}] {h['topic']}: {h.get('overall_stance', '?')} "
                    f"— {h.get('executive_summary', '')[:100]}"
                )
            context += "\n".join(past_lines)

    session = MacroDebateSession(
        session_id=session_id,
        topic=req.topic.title,
        context=context,
        api_key=api_key,
        model=req.model,
        market_context=market_context,
    )

    try:
        orchestrator = _get_macro_orchestrator(session)
        round1, opinions = await orchestrator.run_round1(
            session.topic, session.context,
        )
        session.rounds.append(round1)
        session.opinions = opinions
        session.phase = "round1_done"
    except Exception as e:
        session.error = str(e)

    _macro_sessions[session_id] = session
    if not session.error:
        _save_macro_draft(session)
    return _macro_response(session)


@app.post("/api/macro/{session_id}/round2", response_model=MacroSessionResponse)
async def macro_round2(session_id: str, feedback: FeedbackRequest | None = None):
    """거시 토론 Round 2: 피드백 반영 + 상호 반론."""
    session = _macro_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if session.phase != "round1_done":
        raise HTTPException(status_code=400, detail=f"현재 단계: {session.phase}")

    user_fb = None
    if feedback and feedback.message.strip():
        user_fb = UserFeedback(content=feedback.message.strip())
        session.user_feedbacks.append(user_fb)

    try:
        orchestrator = _get_macro_orchestrator(session)
        round2 = await orchestrator.run_round2(
            session.topic, session.opinions,
            user_feedback=user_fb, context=session.context,
        )
        session.rounds.append(round2)
        session.phase = "round2_done"
        _save_macro_draft(session)
    except Exception as e:
        session.error = str(e)

    return _macro_response(session)


@app.post("/api/macro/{session_id}/round3", response_model=MacroSessionResponse)
async def macro_round3(session_id: str, feedback: FeedbackRequest | None = None):
    """거시 토론 Round 3: 피드백 반영 + 최종 입장."""
    session = _macro_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if session.phase != "round2_done":
        raise HTTPException(status_code=400, detail=f"현재 단계: {session.phase}")

    user_fb = None
    if feedback and feedback.message.strip():
        user_fb = UserFeedback(content=feedback.message.strip())
        session.user_feedbacks.append(user_fb)

    try:
        orchestrator = _get_macro_orchestrator(session)
        round3 = await orchestrator.run_round3(
            session.topic, session.rounds,
            user_feedback=user_fb, context=session.context,
        )
        session.rounds.append(round3)
        session.phase = "round3_done"
        _save_macro_draft(session)
    except Exception as e:
        session.error = str(e)

    return _macro_response(session)


@app.post("/api/macro/{session_id}/synthesize", response_model=MacroSessionResponse)
async def macro_synthesize(session_id: str, feedback: FeedbackRequest | None = None):
    """거시 토론 최종 종합 판정."""
    session = _macro_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    if session.phase != "round3_done":
        raise HTTPException(status_code=400, detail=f"현재 단계: {session.phase}")

    final_comment = ""
    if feedback and feedback.message.strip():
        user_fb = UserFeedback(content=feedback.message.strip())
        session.user_feedbacks.append(user_fb)
        final_comment = feedback.message.strip()

    try:
        orchestrator = _get_macro_orchestrator(session)
        synthesis = await orchestrator.synthesize(
            session.topic, session.rounds,
            session.user_feedbacks, final_comment,
            context=session.context,
        )

        # Parse investment_implications and risk_scenarios
        from src.models.macro import (
            InvestmentImplication,
            RiskScenario,
            ScenarioAnalysisModel,
            ScenarioModel,
            StressTestModel,
        )
        implications = []
        for imp in synthesis.get("investment_implications", []):
            if isinstance(imp, dict):
                implications.append(InvestmentImplication(**imp))
        risk_scenarios = []
        for rs in synthesis.get("risk_scenarios", []):
            if isinstance(rs, dict):
                risk_scenarios.append(RiskScenario(**rs))

        # Parse scenario analysis results
        scenario_analysis = None
        sa_data = synthesis.get("scenario_analysis")
        if sa_data and isinstance(sa_data, dict):
            scenarios = []
            for s in sa_data.get("scenarios", []):
                if isinstance(s, dict):
                    scenarios.append(ScenarioModel(**s))
            stress_tests = []
            for st in sa_data.get("stress_tests", []):
                if isinstance(st, dict):
                    stress_tests.append(StressTestModel(**st))
            scenario_analysis = ScenarioAnalysisModel(
                scenarios=scenarios,
                stress_tests=stress_tests,
                weighted_allocation=sa_data.get("weighted_allocation", {}),
                expected_portfolio_return=sa_data.get("expected_portfolio_return", 0.0),
            )

        result = MacroDebateResult(
            topic=session.topic,
            overall_stance=synthesis.get("overall_stance", "중립"),
            confidence=min(max(synthesis.get("confidence", 0.5), 0.0), 1.0),
            executive_summary=synthesis.get("executive_summary", ""),
            consensus_points=synthesis.get("consensus_points", []),
            dissent_points=synthesis.get("dissent_points", []),
            investment_implications=implications,
            risk_scenarios=risk_scenarios,
            action_items=synthesis.get("action_items", []),
            monitoring_points=synthesis.get("monitoring_points", []),
            rounds=[_round_to_model(r) for r in session.rounds],
            scenario_analysis=scenario_analysis,
        )
        session.result = synthesis
        session.phase = "synthesized"

        # 토론 이력 저장 (영구 저장)
        history_record = {
            "session_id": session.session_id,
            "mode": "macro",
            "topic": session.topic,
            "overall_stance": synthesis.get("overall_stance", ""),
            "executive_summary": synthesis.get("executive_summary", ""),
            "key_insights": synthesis.get("consensus_points", []),
            "dissent_points": synthesis.get("dissent_points", []),
            "confidence": synthesis.get("confidence", 0.5),
            "investment_implications": synthesis.get("investment_implications", []),
            "risk_scenarios": synthesis.get("risk_scenarios", []),
            "action_items": synthesis.get("action_items", []),
            "monitoring_points": synthesis.get("monitoring_points", []),
            "round_count": len(session.rounds),
            "user_feedbacks": [fb.content for fb in session.user_feedbacks],
            "rounds_summary": [
                {
                    "round_number": r.round_number,
                    "messages": [
                        {"agent": m.agent_name, "stance": m.stance, "content": m.content[:300]}
                        for m in r.messages
                    ],
                }
                for r in session.rounds
            ],
        }
        _debate_history_store.append(history_record)
        get_history_store().save(session.session_id, history_record)
        # 완료된 토론의 드래프트 삭제
        get_draft_store().delete(session.session_id)
    except Exception as e:
        session.error = str(e)
        result = None

    return _macro_response(session, result=result)


# ── 토론 이력 (데이터 피드백 루프) ──

_debate_history_store: list[dict] = []  # 과거 토론 기록 저장소


@app.get("/api/macro/{session_id}/history")
async def macro_get_history(session_id: str):
    """거시 토론 전체 기록 조회 (재활용 가능)."""
    session = _macro_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    orchestrator = _get_macro_orchestrator(session)
    history = orchestrator._format_debate_history(session.rounds)
    return {
        "session_id": session_id,
        "topic": session.topic,
        "context": session.context,
        "debate_history": history,
        "user_feedbacks": [{"content": fb.content} for fb in session.user_feedbacks],
        "result": session.result,
    }


@app.get("/api/debate-histories")
async def list_debate_histories():
    """과거 토론 기록 목록 조회 (영구 저장소)."""
    store = get_history_store()
    histories = store.list_all(limit=50)
    return {"histories": histories}


@app.get("/api/debate-histories/{session_id}")
async def get_debate_history(session_id: str):
    """특정 토론 기록 상세 조회."""
    record = get_history_store().load(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="토론 기록을 찾을 수 없습니다.")
    return record


@app.delete("/api/debate-histories/{session_id}")
async def delete_debate_history(session_id: str):
    """토론 기록 삭제."""
    if get_history_store().delete(session_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="토론 기록을 찾을 수 없습니다.")


# ── 중간 저장 / 이어하기 (Draft & Resume) ──


class ResumeRequest(BaseModel):
    api_key: str | None = Field(default=None, description="API 키 (드래프트에 저장된 키가 만료된 경우)")
    model: str | None = Field(default=None, description="모델 오버라이드")


@app.get("/api/drafts")
async def list_drafts():
    """미완료 토론 드래프트 목록."""
    return {"drafts": get_draft_store().list_all()}


@app.get("/api/drafts/{session_id}")
async def get_draft(session_id: str):
    """특정 드래프트 상세 조회."""
    data = get_draft_store().load(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="드래프트를 찾을 수 없습니다.")
    # api_key는 응답에서 제거
    safe = {k: v for k, v in data.items() if k != "api_key"}
    return safe


@app.delete("/api/drafts/{session_id}")
async def delete_draft(session_id: str):
    """드래프트 삭제."""
    # 메모리에서도 제거
    _sessions.pop(session_id, None)
    _macro_sessions.pop(session_id, None)
    if get_draft_store().delete(session_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="드래프트를 찾을 수 없습니다.")


@app.post("/api/debate/resume/{session_id}", response_model=DebateSessionResponse)
async def debate_resume(session_id: str, req: ResumeRequest | None = None):
    """
    중단된 종목 토론을 이어서 진행.

    메모리에 세션이 있으면 그대로 사용하고,
    없으면 드래프트에서 복원한다. 복원된 세션의 현재 phase에서
    다음 라운드를 호출하면 된다.
    """
    session = _sessions.get(session_id)
    if not session:
        data = get_draft_store().load(session_id)
        if not data or data.get("mode") != "stock":
            raise HTTPException(status_code=404, detail="종목 토론 드래프트를 찾을 수 없습니다.")
        session = _deserialize_stock_session(data)
        _sessions[session_id] = session

    # API 키/모델 오버라이드
    if req:
        if req.api_key:
            session.api_key = req.api_key
        if req.model:
            session.model = req.model

    # api_key가 비어있으면 환경변수에서 시도
    if not session.api_key:
        session.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    return DebateSessionResponse(
        session_id=session_id,
        phase=session.phase,
        rounds=session.rounds,
        result=session.result,
        error=session.error,
    )


@app.post("/api/macro/resume/{session_id}", response_model=MacroSessionResponse)
async def macro_resume(session_id: str, req: ResumeRequest | None = None):
    """
    중단된 거시 토론을 이어서 진행.

    메모리에 세션이 있으면 그대로 사용하고,
    없으면 드래프트에서 복원한다.
    """
    session = _macro_sessions.get(session_id)
    if not session:
        data = get_draft_store().load(session_id)
        if not data or data.get("mode") != "macro":
            raise HTTPException(status_code=404, detail="거시 토론 드래프트를 찾을 수 없습니다.")
        session = _deserialize_macro_session(data)
        _macro_sessions[session_id] = session

    # API 키/모델 오버라이드
    if req:
        if req.api_key:
            session.api_key = req.api_key
        if req.model:
            session.model = req.model

    if not session.api_key:
        session.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    result = None
    if session.phase == "synthesized" and session.result:
        # 이미 완료된 세션이면 결과도 포함
        from src.models.macro import MacroDebateResult
        result = MacroDebateResult(
            topic=session.topic,
            rounds=[_round_to_model(r) for r in session.rounds],
        )

    return _macro_response(session, result=result)


# ── PDF/파일 업로드 (영구 저장) ──


@app.post("/api/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    use_vision: bool = Form(False),
    api_key: str = Form(""),
):
    """
    PDF/TXT/CSV 파일을 업로드하여 영구 지식 베이스에 추가.

    업로드된 문서는 이후 모든 토론(종목/거시)에 자동 반영되며
    서버를 재시작해도 유지된다.

    use_vision=True로 설정하면 그래프/차트/모식도를 Claude Vision으로 해석한다.
    Vision 사용 시 api_key가 필요하다.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    content = await file.read()
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="파일 크기는 10MB 이하여야 합니다.")

    try:
        if use_vision and file.filename.lower().endswith(".pdf"):
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not resolved_key:
                raise HTTPException(
                    status_code=400,
                    detail="Vision 분석에는 ANTHROPIC_API_KEY가 필요합니다.",
                )
            client = ClaudeClient(api_key=resolved_key)
            extracted = await extract_text_from_file_with_vision(content, file.filename, client)
        else:
            extracted = extract_text_from_file(content, file.filename)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"파일 처리 실패: {e}")

    # 영구 저장
    try:
        entry = get_document_store().add(file.filename, extracted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 저장 실패: {e}")

    return {
        "filename": file.filename,
        "doc_id": entry["doc_id"],
        "text_length": len(extracted),
        "vision_used": use_vision and file.filename.lower().endswith(".pdf"),
        "preview": extracted[:500] + "..." if len(extracted) > 500 else extracted,
    }


@app.get("/api/upload/documents")
async def list_uploaded_documents():
    """영구 저장된 문서 목록 조회."""
    return {"documents": get_document_store().list_all()}


@app.delete("/api/upload/documents/{doc_id}")
async def delete_uploaded_document(doc_id: str):
    """특정 문서 삭제."""
    if get_document_store().delete(doc_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")


@app.delete("/api/upload/documents")
async def clear_uploaded_documents():
    """업로드된 모든 문서 삭제."""
    store = get_document_store()
    for doc in store.list_all():
        store.delete(doc["doc_id"])
    return {"cleared": True}


@app.get("/api/market-data")
async def get_market_data(ticker: str | None = None, name: str | None = None):
    """실시간 시장 데이터 조회."""
    try:
        snapshot = await fetch_market_snapshot(
            ticker=ticker, ticker_name=name,
            include_commodities=True,
        )
        return {
            "timestamp": snapshot.timestamp,
            "indices": [{"name": i.name, "value": i.value, "change": i.change, "change_pct": i.change_pct} for i in snapshot.indices],
            "exchange_rates": [{"name": f.name, "value": f.value, "change": f.change, "change_pct": f.change_pct} for f in snapshot.exchange_rates],
            "quotes": [{"ticker": q.ticker, "name": q.name, "price": q.price, "change": q.change, "change_pct": q.change_pct} for q in snapshot.quotes],
            "news": [{"title": n.title, "source": n.source} for n in snapshot.news],
            "has_data": snapshot.has_data,
        }
    except Exception as e:
        return {"error": str(e), "has_data": False}


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


# ── 설정 관리 API ──

@app.get("/api/settings")
async def get_current_settings():
    """현재 설정 조회."""
    return get_settings().to_dict()


@app.patch("/api/settings")
async def patch_settings(updates: dict):
    """설정 업데이트. 변경할 필드만 전달."""
    settings = update_settings(**updates)
    return settings.to_dict()


@app.get("/api/macro/{session_id}/usage")
async def macro_usage(session_id: str):
    """세션의 API 사용량/비용 조회."""
    session = _macro_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return session.usage_tracker.summary()


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
