# LinkedIn post raw material — Agentic Web Builder

Source: repository `Agentic-Web Builder` (README, `docker-compose.yml`, `nginx.conf`, GitHub Actions `production.yml`, `backend/agent/graph.py`, Dockerfiles, frontend API client).

---

## 1. Project overview

**What it does (1–2 lines)**  
Takes one natural-language prompt and returns a full multi-file web project as structured JSON from a FastAPI backend; the UI renders a file tree and syntax-highlighted sources and lets the user download everything as a ZIP in the browser—no server-side project storage.

**Problem it solves**  
Single-shot LLM “generate my whole app” flows often truncate, break imports, or return unusable free-form JSON. This system forces the model through **Planner → Architect → Coder** stages with **Pydantic-shaped structured outputs** so the contract with the frontend stays stable.

**Target user**  
Developers and builders who want **fast, disposable scaffolding** (prototypes, homework-style apps, boilerplate) without maintaining file storage or a database for generated artifacts. Product-oriented users who want a **working file tree** they can inspect and export immediately.

---

## 2. Core features (differentiating only)

1. **LangGraph multi-agent pipeline** — Sequential graph: `planner` → `architect` → `coder`, compiled state machine with conditional edge from coder to `END` (not a single monolithic prompt).
2. **Schema-bound generation** — `Plan`, `TaskPlan`, `CoderOutput` (Pydantic) consumed via LangChain `with_structured_output(..., method="json_mode")` on **ChatGroq**, avoiding brittle “return JSON” strings and prior tool-calling failure modes.
3. **Per-stage temperature policy** — Planner 0.7, Architect 0.2, Coder 0.0 (explicit trade-off: ideation vs. deterministic code-shaped output).
4. **Stateless backend** — Generated files live in memory in `CoderState`; API returns `{ success, files: [{ path, content }] }` only; no DB for generations in the core flow described in README.
5. **Client-side ZIP export** — JSZip + FileSaver in the browser; shifts bundling and storage cost off the server (README rationale: no disk cleanup, simpler ops).
6. **In-UI exploration** — File explorer + code viewer (React) for immediate review before download; generation request is abortable (`AbortController` in route code).

*Secondary (template-level):* Frontend includes Supabase-powered auth UI (`useAuth`, login/signup/OAuth routes); this is orthogonal to the generation API but exists in the shipped app.

---

## 3. Tech stack (specific)

| Layer | Stack |
|--------|--------|
| **Frontend** | React 19, TypeScript, Vite 7, TanStack Router / React Start, Tailwind CSS 4, Radix/shadcn-style UI, react-syntax-highlighter, axios/fetch, JSZip, file-saver. Vitest for tests. |
| **Backend** | Python 3.10, FastAPI, Uvicorn, Pydantic, LangChain + LangGraph, `langchain-groq` (`ChatGroq`), `python-dotenv`, `httpx`. Pytest for tests. |
| **AI / ML** | **Groq** inference API; model selected via **`MODEL` environment variable** (CI pins `llama-3.3-70b-versatile`; README documents same family for speed on chained calls). Structured output via **JSON mode** + Pydantic schemas, not tool calls for file emission. |
| **Database** | **None** for the core generate/export path. Optional **Supabase** client for frontend authentication (env: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`). |
| **Infrastructure / deployment** | **DigitalOcean droplet** (SSH deploy). **Docker Compose** orchestrates three services: `backend` (image built from `backend/`), `frontend` (image built from `frontend/`), `nginx` (`nginx:latest`). **GitHub Actions** (`production.yml`): on push to `main`, run backend pytest + frontend vitest, then **appleboy/ssh-action** to droplet: `cd ~/repo`, `git pull`, `docker compose down`, `docker compose up --build -d`. Secrets: `DROPLET_IP`, `SSH_PRIVATE_KEY`. Compose passes `GROQ_API_KEY`, `MODEL` to backend. |
| **APIs / integrations** | **Groq** (LLM). **Supabase Auth** (OAuth helpers in `frontend/src/lib/supabase.ts`). Frontend calls **`POST /generate`** (JSON body `{ "prompt": string }`). |

---

## 4. Architecture / system design

**End-to-end**  
Browser collects prompt → `POST` to API base + `/generate` → FastAPI validates body → `generate_codebase()` runs compiled LangGraph with initial state `{ user_prompt, debug }` → planner produces `Plan` → architect produces `TaskPlan` with `implementation_steps` → coder invokes structured LLM once with combined plan + task plan JSON in user message, fills `CoderState.generated_files`, marks `DONE` → API returns file list or 422/500 with error payload.

**Patterns**  
- **Directed agent graph** (LangGraph), not microservices.  
- **Synchronous request/response** from the HTTP client’s perspective (one user action drives the full chain; `recursion_limit` 100 on invoke).  
- **Reverse proxy as API gateway**: nginx terminates HTTP on port 80 and splits traffic.

**Scalability / performance (documented intent)**  
- Stateless API + in-memory generation → horizontal scaling story is “add more API workers” without shared filesystem for zips.  
- Groq chosen for **low latency** across multiple LLM round-trips in one user session (README).  
- **No server-side ZIP or object storage** reduces I/O and operational surface area.

---

## 5. AI component (critical)

**Capabilities**  
1. **Planning** — Structured app spec: name, description, tech stack, feature list, planned files with purposes.  
2. **Architecture** — Decomposes plan into ordered `implementation_steps` (filepath + task description).  
3. **Coding** — Emits `CoderOutput`: list of `FileItem` `{ path, content }` for the whole codebase in one structured response (coder advances `current_step_idx` to end after success).

**Models / frameworks**  
- **Frameworks:** LangGraph state graph, LangChain structured outputs, Pydantic v2 models in `backend/agent/states.py`.  
- **Model:** Configurable **`MODEL` env**; repository and CI reference **`llama-3.3-70b-versatile`** via Groq.

**What makes the integration useful**  
- **Decomposition beats one-shot codegen** for multi-file coherence.  
- **json_mode + Pydantic** is the explicit fix for tool-calling failures and markdown-wrapped JSON (README + `graph.py` implementation).  
- **Temperature split** encodes engineering judgment: creativity early, strict formatting late.

---

## 6. Challenges faced (concrete)

1. **`tool_use_failed` / tool-calling brittleness** — Large code payloads inside tool arguments caused failures; pipeline was refactored to **native structured JSON output** instead of virtual “write file” tools (README).  
2. **Malformed or decorated JSON** — Models prefixing ```json or drifting from schema broke consumers; **structured output + schema** replaced “please return JSON” prompting (README).  
3. **Architect empty steps** — Runtime guard: `ValueError` if `implementation_steps` is empty (`graph.py`).  
4. **Coder failures** — Try/except around coder invoke; errors folded into response shape with `"Code generation failed"` and exception details for debugging (`graph.py`).  
5. **Deployment wiring** — Single entry via nginx: must align **rewrite** `/api/*` → backend `/*` with frontend `VITE_API_BASE` (e.g. `https://gauravkeshri.dev/api` in `frontend/Dockerfile`) so `/generate` hits the backend correctly. *Note:* `docker-compose.yml` shows a different `VITE_API_BASE` string than the Dockerfile; production truth should be unified to avoid misbuilt images.

---

## 7. Key learnings

1. **Schema-driven contracts beat prose-only prompts** for multi-file codegen reliability.  
2. **Multi-stage graphs** turn an ill-posed “build everything” problem into bounded subproblems (plan → steps → files).  
3. **Push complexity to the client** (ZIP in browser) when artifacts are disposable—simpler backend and cheaper hosting footprint.  
4. **CI-gated deploys** (`needs: test`) prevent shipping broken graphs/API regressions when tests exist and are maintained.  
5. **Observable stages in UI** (loading steps mirroring planner/architect/coder) help perceived performance even when the backend is one long request.

---

## 8. Performance / impact

**No committed production metrics** (latency p95, cost per generation, token counts) in repo.

**Practical benefits (evidence-based)**  
- Fewer classes of serialization errors for the frontend (`GenerateResponse` validation in `api.ts`).  
- Reduced operational burden: no DB migrations for generated projects, no blob lifecycle policies for ZIPs on server.  
- Faster iteration for users who only need downloadable source, not a hosted runtime of the generated app.

---

## 9. Deployment details (droplet)

**Host**  
DigitalOcean VM; deploy user `root`; app directory `~/repo` on the server (workflow script).

**How it runs**  
- **`docker compose up --build -d`** after `git pull` — builds images and runs stack detached.  
- **nginx** container: host **`80:80`**, mounts `./nginx.conf` read-only.  
- **Upstream routing** (`nginx.conf`):  
  - `location /api/` → `proxy_pass http://backend:8000` with `rewrite ^/api/(.*) /$1 break;` so external `https://<host>/api/generate` maps to backend `/generate`.  
  - `location /` → `proxy_pass http://frontend:8080`.  
- **Backend container:** `python:3.10-slim`, `uvicorn api:app --host 0.0.0.0 --port 8000` (exposed internally as 8000).  
- **Frontend container:** `node:24-slim`, `npm run build` with baked `VITE_API_BASE`, then **`wrangler dev --ip 0.0.0.0 --port 8080`** to serve the built app inside Docker (unusual pattern: Cloudflare Wrangler as static server behind nginx).

**Secrets / env**  
- Droplet must have `GROQ_API_KEY` and `MODEL` available to Compose (from `.env` or host env when compose runs).  
- TLS/HTTPS: not defined in-repo for nginx (only `listen 80`); if the public URL is HTTPS, termination may be outside this file (e.g. DO load balancer or external proxy)—call that out honestly on LinkedIn if you use it.

**Scaling / optimization**  
- Single-node Compose; scale-out would mean multiple droplets or migrating to orchestrator + external Groq key rotation, not implemented here.  
- Image layer caching: Dockerfiles copy `requirements.txt` / `package*.json` before full source copy.

---

## 10. Demo / usage

**How to use (production-shaped)**  
1. Open the deployed site (domain referenced in Docker build: **`https://gauravkeshri.dev`** — verify HTTPS and current availability before posting).  
2. Complete auth if the home route requires Supabase session (code paths: `useAuth` on `/`).  
3. Enter a prompt describing the web app to scaffold.  
4. Wait for generation; browse files in the UI; download ZIP.

**API-only usage**  
```http
POST https://<your-host>/api/generate
Content-Type: application/json

{"prompt": "Create a React todo app with local storage"}
```

**Example use-case**  
“Spin up a small static or React-style toy app with correct folder layout and multiple files, then download the repo as ZIP for local `npm install` or editing.”

**Live link**  
- **`https://gauravkeshri.dev`** (from `frontend/Dockerfile` / nginx routing assumptions). *Confirm live before publishing.*

---

## 11. One-liner hook options (openings)

1. **I stopped asking one LLM call to “build the whole app” and chained three schema-locked agents on Groq—here’s why the JSON finally stopped breaking.**  
2. **Your codegen API doesn’t need a database or disk if you treat the browser as the artifact store—this is how we ship multi-file ZIPs from a $5 droplet.**  
3. **We deleted tool-calling for file writes and used Pydantic + json_mode instead—and the `tool_use_failed` errors disappeared.**

---

*End of extracted material. Trim or merge sections when drafting the final LinkedIn post.*
