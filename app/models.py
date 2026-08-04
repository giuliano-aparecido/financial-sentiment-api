from pydantic import BaseModel


class QueryRequest(BaseModel):
    user_query: str


class UpdateInferenceURLRequest(BaseModel):
    url: str
