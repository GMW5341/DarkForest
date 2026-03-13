"""
DarkForest Web Server — FastAPI 기반 웹 API.

Multi-Agent Debate 포트폴리오 분석을 REST API로 제공한다.
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
from src.models.analysis import PortfolioAnalysis
from src.models.portfolio import Portfolio


# ── In-memory job store (분석은 오래 걸리므로 비동기 작업으로 처리) ──

class AnalysisJob(BaseModel):
    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    result: PortfolioAnalysis | None = None
    error: str | None = None


_jobs: dict[str, AnalysisJob] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="DarkForest",
    description="Multi-Agent Debate 투자 포트폴리오 분석기",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ──

class AnalyzeRequest(BaseModel):
    portfolio: Portfolio
    api_key: str | None = Field(default=None, description="Anthropic API 키 (선택)")
    model: str = Field(default="claude-sonnet-4-20250514")
    debate_only: bool = Field(
        default=False,
        description="True면 토론만 실행 (빠른 모드)",
    )


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
    """포트폴리오 분석을 시작한다 (비동기)."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY가 필요합니다. "
                   "요청 본문의 api_key 또는 서버 환경변수로 설정하세요.",
        )

    job_id = str(uuid.uuid4())[:8]
    job = AnalysisJob(job_id=job_id, status="running")
    _jobs[job_id] = job

    import asyncio
    asyncio.create_task(
        _run_analysis(job, req.portfolio, api_key, req.model, req.debate_only)
    )

    return AnalyzeResponse(job_id=job_id, status="running")


async def _run_analysis(
    job: AnalysisJob,
    portfolio: Portfolio,
    api_key: str,
    model: str,
    debate_only: bool,
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
    """분석 작업 상태를 조회한다."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )


@app.post("/api/analyze/sync", response_model=PortfolioAnalysis)
async def analyze_sync(req: AnalyzeRequest):
    """포트폴리오 분석을 동기적으로 실행하고 결과를 바로 반환한다."""
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="ANTHROPIC_API_KEY가 필요합니다.",
        )

    try:
        analyzer = PortfolioAnalyzer(api_key=api_key, model=req.model)
        if req.debate_only:
            return await analyzer.debate(req.portfolio)
        return await analyzer.analyze(req.portfolio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "darkforest", "version": "0.2.0"}


# ── Static files (프론트엔드) ──
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "DarkForest API is running. POST /api/analyze to start."}


# Mount static files if directory exists
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
