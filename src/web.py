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

        # 토론 이력 저장 (피드백 루프용)
        _debate_history_store.append({
            "session_id": session_id,
            "mode": "stock",
            "topic": f"{session.holding.name} ({session.holding.ticker})",
            "overall_stance": synthesis.get("final_decision", ""),
            "executive_summary": synthesis.get("final_reasoning", ""),
            "key_insights": synthesis.get("consensus_points", []),
        })
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
    ):
        self.session_id = session_id
        self.topic = topic
        self.context = context
        self.api_key = api_key
        self.model = model
        self.phase = "waiting_start"
        self.rounds: list[MacroDebateRound] = []
        self.opinions: list[MacroAgentOpinion] = []
        self.user_feedbacks: list[UserFeedback] = []
        self.result: dict | None = None
        self.error: str | None = None


_macro_sessions: dict[str, MacroDebateSession] = {}


class MacroStartRequest(BaseModel):
    topic: MacroTopic
    api_key: str | None = Field(default=None)
    model: str = Field(default="claude-sonnet-4-20250514")
    include_past_insights: bool = Field(
        default=True, description="과거 토론 인사이트를 배경 정보에 포함할지 여부"
    )


class MacroSessionResponse(BaseModel):
    session_id: str
    phase: str
    rounds: list[MacroDebateRoundModel] = Field(default_factory=list)
    result: MacroDebateResult | None = None
    error: str | None = None


def _get_macro_orchestrator(session: MacroDebateSession) -> MacroDebateOrchestrator:
    client = ClaudeClient(api_key=session.api_key, model=session.model)
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
    )


@app.post("/api/macro/start", response_model=MacroSessionResponse)
async def macro_start(req: MacroStartRequest):
    """거시 경제 토론 Round 1 시작."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY가 필요합니다.")

    # 과거 토론 인사이트 주입
    context = req.topic.context
    if req.include_past_insights and _debate_history_store:
        past_lines = ["\n\n## 과거 토론 인사이트 (참고용)"]
        for h in _debate_history_store[-5:]:  # 최근 5개만
            past_lines.append(
                f"- [{h['mode'].upper()}] {h['topic']}: {h.get('overall_stance', '?')} "
                f"— {h.get('executive_summary', '')[:100]}"
            )
        context += "\n".join(past_lines)

    session_id = str(uuid.uuid4())[:8]
    session = MacroDebateSession(
        session_id=session_id,
        topic=req.topic.title,
        context=context,
        api_key=api_key,
        model=req.model,
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

        # 토론 이력 저장 (피드백 루프용)
        _debate_history_store.append({
            "session_id": session.session_id,
            "mode": "macro",
            "topic": session.topic,
            "overall_stance": synthesis.get("overall_stance", ""),
            "executive_summary": synthesis.get("executive_summary", ""),
            "key_insights": synthesis.get("consensus_points", []),
        })
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
    """과거 토론 기록 목록 조회."""
    return {"histories": _debate_history_store}


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
