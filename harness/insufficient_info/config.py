from pydantic import BaseModel, Field


class JudgeConfig(BaseModel):
    min_max_score: float = Field(default=0.22, description="Минимальный лучший score retrieval")
    min_mean_score: float = Field(default=0.12, description="Средний score по top-k")
    min_chunks: int = Field(default=1, description="Минимум чанков для ALLOW")


DEFAULT_JUDGE_CONFIG = JudgeConfig()
