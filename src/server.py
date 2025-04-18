from fastapi import FastAPI, Request
from pydantic import BaseModel
from embed import SentenceTransformer, MODEL_NAME
from rag_chain import run_rag_pipeline

app = FastAPI()
model = SentenceTransformer(MODEL_NAME)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    answer = run_rag_pipeline(query.question, model)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)