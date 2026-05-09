from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent.graph import generate_codebase


class PromptRequest(BaseModel):
    prompt: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # allows all origins
    allow_credentials=False,
    allow_methods=["*"], # allows all methods 
    allow_headers=["*"], # allows all headers
)

def _error_body(error: str, details: str = "") -> dict:
    return {"success": False, "error": error, "details": details}


@app.post("/generate")
async def generate(request: PromptRequest):
    try:
        result = await generate_codebase(request.prompt)

        if not result.get("success"):
            return JSONResponse(
                status_code=422,
                content=_error_body(
                    result.get("error") or "Code generation failed",
                    result.get("details") or "",
                ),
            )

        coder_state = result.get("coder_state")

        if not coder_state:
            return JSONResponse(
                status_code=422,
                content=_error_body("Missing coder state", ""),
            )

        if isinstance(coder_state, dict):
            files = coder_state.get("generated_files")
        else:
            files = coder_state.generated_files

        if files is None:
            files = []

        if not files:
            return JSONResponse(
                status_code=422,
                content=_error_body("No files generated", ""),
            )

        print("==== FINAL FILES ====")
        print(files)

        return {"success": True, "files": files}

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=_error_body("Internal server error", str(exc)),
        )
