import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


class PromptRequest(BaseModel):
    prompt: str


@app.post("/")
async def analyze_fanout(data: PromptRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=data.prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )

        meta = response.candidates[0].grounding_metadata
        queries = (
            meta.web_search_queries
            if meta and meta.web_search_queries
            else []
        )

        sources = []
        if meta and meta.grounding_chunks:
            for chunk in meta.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    sources.append(
                        {"title": chunk.web.title, "uri": chunk.web.uri}
                    )

        return {
            "answer": response.text,
            "fanout_queries": list(queries),
            "sources": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))