from src.engine.question_frame import QuestionFrame, FrameCategory


def __getattr__(name: str):
    """Lazy import to avoid circular dependency with agents.debate."""
    if name == "ReasoningEngine":
        from src.engine.reasoning import ReasoningEngine
        return ReasoningEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["QuestionFrame", "FrameCategory", "ReasoningEngine"]
