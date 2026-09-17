from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_query: str = Field(min_length=1, max_length=2000)


class UpdateInferenceURLRequest(BaseModel):
    url: str
    model: str | None = None


class UpdateYfCrumbRequest(BaseModel):
    crumb: str
    cookies: dict[str, str]
