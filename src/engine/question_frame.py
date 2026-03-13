"""
Question Frame Engine — 질문 프레임 시스템.

'결론이 100% 맞을 수는 없어도, 분석이 100% 맞지 않으면 안 된다.'

이 모듈은 기업/종목 분석을 위한 구조화된 질문 프레임을 정의한다.
각 프레임은 특정 시점에 특정 논리와 사고를 근거로 판단을 내리기 위한
확고한 관점(frame)을 제공한다.

단순한 지식의 전달이 아닌, 사고 훈련의 장으로서 기능한다.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class FrameCategory(str, Enum):
    """질문 프레임 카테고리."""
    BUSINESS_IDENTITY = "business_identity"       # 사업 정체성
    REVENUE_STRUCTURE = "revenue_structure"        # 매출 구조
    COMPETITIVE_MOAT = "competitive_moat"          # 경쟁 해자
    FINANCIAL_HEALTH = "financial_health"          # 재무 건전성
    VALUATION = "valuation"                        # 가치 평가
    CATALYST = "catalyst"                          # 촉매/이벤트
    RISK = "risk"                                  # 리스크
    PORTFOLIO_FIT = "portfolio_fit"                # 포트폴리오 적합성


@dataclass
class QuestionFrame:
    """
    하나의 질문 프레임.

    질문 프레임은 '매수/매도' 결론을 내리기 위한 것이 아니라,
    판단 당시의 논리 정합성을 100% 보장하기 위한 사고 체계이다.
    """
    category: FrameCategory
    name: str
    core_question: str  # 핵심 질문 — 이 프레임이 답해야 할 단 하나의 질문
    sub_questions: list[str] = field(default_factory=list)  # 핵심 질문을 분해한 하위 질문
    evaluation_criteria: list[str] = field(default_factory=list)  # 판단 기준
    red_flags: list[str] = field(default_factory=list)  # 경고 신호
    prompt_template: str = ""  # Claude에 전달할 프롬프트 템플릿


# ──────────────────────────────────────────────────────────────
# 기본 제공 질문 프레임들
# ──────────────────────────────────────────────────────────────

DEFAULT_FRAMES: list[QuestionFrame] = [
    # ① 사업 정체성
    QuestionFrame(
        category=FrameCategory.BUSINESS_IDENTITY,
        name="사업 정체성 프레임",
        core_question="이 회사는 본질적으로 무엇을 파는 회사인가?",
        sub_questions=[
            "이 회사의 제품/서비스를 한 문장으로 설명할 수 있는가?",
            "이 회사가 없어지면 고객은 어디로 가는가?",
            "5년 후에도 같은 사업을 하고 있을 것인가?",
        ],
        evaluation_criteria=[
            "사업 모델을 명확히 설명할 수 있는지 여부",
            "핵심 가치 제안(value proposition)이 분명한지 여부",
            "사업의 지속 가능성에 대한 논리적 근거",
        ],
        red_flags=[
            "사업 모델을 한 문장으로 설명할 수 없음",
            "핵심 사업이 무엇인지 경영진 발언이 매번 달라짐",
            "트렌드 의존적 사업 (본질 없는 테마주)",
        ],
        prompt_template=(
            "다음 종목에 대해 '사업 정체성' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 이 회사는 본질적으로 무엇을 파는 회사인가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 이 회사의 제품/서비스를 한 문장으로 설명할 수 있는가?\n"
            "2. 이 회사가 없어지면 고객은 어디로 가는가?\n"
            "3. 5년 후에도 같은 사업을 하고 있을 것인가?\n\n"
            "경고 신호가 있다면 명시해주세요.\n"
            "분석의 논리적 근거를 반드시 제시하세요. "
            "결론보다 과정의 논리 정합성이 중요합니다."
        ),
    ),

    # ② 매출 구조
    QuestionFrame(
        category=FrameCategory.REVENUE_STRUCTURE,
        name="매출 구조 프레임",
        core_question="매출은 어디서, 어떻게 발생하며, 그 구조는 건강한가?",
        sub_questions=[
            "매출의 Top 3 소스는 무엇이며 각각의 비중은?",
            "매출 집중도가 과도하지 않은가? (특정 고객/제품 의존도)",
            "반복 매출(recurring revenue) 비중은 어느 정도인가?",
            "매출 성장의 드라이버는 가격인가, 볼륨인가?",
        ],
        evaluation_criteria=[
            "매출원의 다변화 정도",
            "반복 매출 비중 (높을수록 긍정적)",
            "매출 성장률 vs 업종 평균",
            "매출 구조의 변화 추이",
        ],
        red_flags=[
            "단일 고객 매출 의존도 30% 이상",
            "매출은 성장하지만 이익은 정체",
            "일회성 매출이 반복적으로 포함됨",
        ],
        prompt_template=(
            "다음 종목에 대해 '매출 구조' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 매출은 어디서, 어떻게 발생하며, 그 구조는 건강한가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 매출의 Top 3 소스는 무엇이며 각각의 비중은?\n"
            "2. 매출 집중도가 과도하지 않은가?\n"
            "3. 반복 매출(recurring revenue) 비중은 어느 정도인가?\n"
            "4. 매출 성장의 드라이버는 가격인가, 볼륨인가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),

    # ③ 경쟁 해자
    QuestionFrame(
        category=FrameCategory.COMPETITIVE_MOAT,
        name="경쟁 해자 프레임",
        core_question="경쟁자가 이 회사를 모방하기 어려운 이유는 무엇인가?",
        sub_questions=[
            "이 회사만의 진입 장벽은 무엇인가? (기술, 규모, 네트워크, 브랜드, 규제)",
            "전환 비용(switching cost)은 얼마나 높은가?",
            "시간이 지날수록 해자가 강해지는가, 약해지는가?",
        ],
        evaluation_criteria=[
            "해자의 종류와 강도를 구체적으로 설명할 수 있는지",
            "해자가 실적으로 증명되고 있는지 (마진 유지/확대)",
            "해자의 지속 가능성",
        ],
        red_flags=[
            "해자라고 주장하지만 마진이 하락 추세",
            "기술적 해자가 빠르게 무력화될 수 있는 산업",
            "규제 해자에만 의존 (규제 변화 리스크)",
        ],
        prompt_template=(
            "다음 종목에 대해 '경쟁 해자' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 경쟁자가 이 회사를 모방하기 어려운 이유는 무엇인가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 이 회사만의 진입 장벽은 무엇인가?\n"
            "2. 전환 비용(switching cost)은 얼마나 높은가?\n"
            "3. 시간이 지날수록 해자가 강해지는가, 약해지는가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),

    # ④ 재무 건전성
    QuestionFrame(
        category=FrameCategory.FINANCIAL_HEALTH,
        name="재무 건전성 프레임",
        core_question="이 회사는 재무적으로 지속 가능한 상태인가?",
        sub_questions=[
            "부채비율은 업종 평균 대비 적정한가?",
            "영업현금흐름은 꾸준히 플러스인가?",
            "이익의 질(quality of earnings)은 높은가? (영업이익 vs 영업현금흐름)",
            "자본 배분 정책은 합리적인가? (배당, 자사주, 재투자)",
        ],
        evaluation_criteria=[
            "부채비율 및 이자보상배율 적정 여부",
            "FCF(Free Cash Flow) 창출 능력",
            "ROE/ROIC 추이",
            "운전자본 관리 효율성",
        ],
        red_flags=[
            "영업이익은 흑자이나 영업현금흐름이 적자",
            "부채가 급격히 증가하는 추세",
            "매출채권 회전일수가 계속 늘어남",
            "감가상각 정책 변경 등 회계적 이익 관리 의심",
        ],
        prompt_template=(
            "다음 종목에 대해 '재무 건전성' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 이 회사는 재무적으로 지속 가능한 상태인가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 부채비율은 업종 평균 대비 적정한가?\n"
            "2. 영업현금흐름은 꾸준히 플러스인가?\n"
            "3. 이익의 질(quality of earnings)은 높은가?\n"
            "4. 자본 배분 정책은 합리적인가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),

    # ⑤ 밸류에이션
    QuestionFrame(
        category=FrameCategory.VALUATION,
        name="가치 평가 프레임",
        core_question="현재 가격은 이 회사의 본질 가치 대비 합리적인가?",
        sub_questions=[
            "현재 PER/PBR은 과거 밴드 대비 어디에 위치하는가?",
            "동종 업계 대비 프리미엄/디스카운트는 정당화되는가?",
            "시장이 이 회사에 부여한 기대는 현실적인가?",
        ],
        evaluation_criteria=[
            "절대적 밸류에이션 (DCF 관점의 합리성)",
            "상대적 밸류에이션 (Peer 대비 위치)",
            "기대 이익 성장률 대비 가격의 합리성",
        ],
        red_flags=[
            "실적 대비 밸류에이션이 역사적 고점",
            "성장률은 둔화되는데 성장주 밸류에이션 유지",
            "'미래 가치' 논리로만 현재 가격 정당화",
        ],
        prompt_template=(
            "다음 종목에 대해 '가치 평가' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 현재 가격은 이 회사의 본질 가치 대비 합리적인가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 현재 PER/PBR은 과거 밴드 대비 어디에 위치하는가?\n"
            "2. 동종 업계 대비 프리미엄/디스카운트는 정당화되는가?\n"
            "3. 시장이 이 회사에 부여한 기대는 현실적인가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),

    # ⑥ 촉매/이벤트
    QuestionFrame(
        category=FrameCategory.CATALYST,
        name="촉매 프레임",
        core_question="가격을 움직일 구체적인 이벤트나 촉매가 있는가?",
        sub_questions=[
            "향후 6~12개월 내 주가에 영향을 줄 이벤트는 무엇인가?",
            "그 이벤트의 발생 확률과 영향 크기는?",
            "시장은 이 촉매를 이미 가격에 반영했는가?",
        ],
        evaluation_criteria=[
            "촉매의 구체성 (구체적 일정과 크기가 있는가)",
            "촉매의 실현 가능성",
            "선반영 여부에 대한 판단",
        ],
        red_flags=[
            "'언젠가는' 식의 불확실한 촉매에만 의존",
            "촉매가 이미 주가에 완전히 반영된 상태",
            "부정적 촉매(규제, 소송 등)를 간과",
        ],
        prompt_template=(
            "다음 종목에 대해 '촉매' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 가격을 움직일 구체적인 이벤트나 촉매가 있는가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 향후 6~12개월 내 주가에 영향을 줄 이벤트는?\n"
            "2. 그 이벤트의 발생 확률과 영향 크기는?\n"
            "3. 시장은 이 촉매를 이미 가격에 반영했는가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),

    # ⑦ 리스크
    QuestionFrame(
        category=FrameCategory.RISK,
        name="리스크 프레임",
        core_question="이 투자가 실패하는 시나리오는 무엇인가?",
        sub_questions=[
            "최악의 시나리오에서 최대 손실은 얼마인가?",
            "이 투자의 가장 큰 리스크 3가지는?",
            "그 리스크를 관리하거나 헤지할 수 있는 방법은?",
            "내가 틀렸다면, 어떤 신호에서 손절해야 하는가?",
        ],
        evaluation_criteria=[
            "리스크를 구체적으로 식별하고 정량화할 수 있는지",
            "손절 기준이 명확히 설정되어 있는지",
            "리스크 대비 수익(Risk-Reward)이 합리적인지",
        ],
        red_flags=[
            "리스크 요인을 하나도 제시하지 못함 (과신)",
            "손절 기준 없이 '장기 투자' 논리에만 의존",
            "하나의 종목에 포트폴리오의 30% 이상 집중",
        ],
        prompt_template=(
            "다음 종목에 대해 '리스크' 관점에서 분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "핵심 질문: 이 투자가 실패하는 시나리오는 무엇인가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 최악의 시나리오에서 최대 손실은?\n"
            "2. 이 투자의 가장 큰 리스크 3가지는?\n"
            "3. 그 리스크를 관리하거나 헤지할 수 있는 방법은?\n"
            "4. 틀렸다면 어떤 신호에서 손절해야 하는가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),

    # ⑧ 포트폴리오 적합성
    QuestionFrame(
        category=FrameCategory.PORTFOLIO_FIT,
        name="포트폴리오 적합성 프레임",
        core_question="이 종목은 내 포트폴리오 전체 관점에서 적합한가?",
        sub_questions=[
            "이 종목을 추가/유지함으로써 포트폴리오 분산이 개선되는가?",
            "기존 보유 종목과의 상관관계는 어떠한가?",
            "이 종목의 비중은 내 위험 성향에 부합하는가?",
            "이 종목 대신 더 나은 대안이 있는가?",
        ],
        evaluation_criteria=[
            "포트폴리오 전체 분산 효과 기여도",
            "기존 포지션과의 중복/상관관계",
            "투자 목표와의 정렬도",
        ],
        red_flags=[
            "같은 섹터에 과도하게 집중",
            "상관관계가 높은 종목끼리 비중이 큼",
            "투자 목표와 일치하지 않는 종목 보유",
        ],
        prompt_template=(
            "다음 종목과 포트폴리오에 대해 '포트폴리오 적합성' 관점에서 "
            "분석해주세요.\n\n"
            "종목 정보:\n{holding_info}\n\n"
            "포트폴리오 전체 구성:\n{portfolio_summary}\n\n"
            "핵심 질문: 이 종목은 포트폴리오 전체 관점에서 적합한가?\n\n"
            "다음 하위 질문들에 답해주세요:\n"
            "1. 포트폴리오 분산이 개선되는가?\n"
            "2. 기존 보유 종목과의 상관관계는?\n"
            "3. 비중은 위험 성향에 부합하는가?\n"
            "4. 더 나은 대안이 있는가?\n\n"
            "경고 신호가 있다면 명시하고, 분석의 논리적 근거를 제시하세요."
        ),
    ),
]
