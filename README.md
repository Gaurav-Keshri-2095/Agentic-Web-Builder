# Agentic Web Builder

**A Generative Multi-Agent Architecture for Rapid Web Application Scaffolding**

---

## 1. Project Overview

Agentic Web Builder is an experimental, end-to-end AI platform that turns a single natural language prompt into a complete, structured web application codebase. 

### The Problem It Solves
Building a web project from zero usually requires scaffolding—setting up the folder structure, configuring routers, wiring up mocked APIs, and writing boilerplate styles. Single-prompt AI interfaces (like standard ChatGPT) hallucinate badly when asked to generate multi-file projects, frequently truncating files, missing imports, or breaking the frontend/backend contract.

### Why This Project Exists
This project was built to implement a **schema-driven, multi-agent AI pipeline** that forces Large Language Models to think methodically. By decoupling the generation process into planning, architecting, and coding stages—each strictly bound to Pydantic schemas—the system dramatically reduces hallucinations and outputs syntactically correct, ready-to-download projects.

### Who This Is Useful For
- **Product Managers & Founders:** Quickly mock up functional prototypes.
- **Frontend/Backend Developers:** Skip boilerplate and immediately jump into feature implementation.
- **AI Enthusiasts:** Learn how multi-agent LangGraph architectures enforce structured outputs.

---

## 2. High-Level Architecture

The system follows a stateless, memory-efficient pattern. The entire generation process happens in memory and flows efficiently from the browser, through the FastAPI layer, into the LangGraph state machine, and back to the user without ever touching the disk.

### The Flow
1. **User Request**: User enters a prompt in the React frontend.
2. **API Layer**: Frontend POSTs to the backend `/generate` endpoint.
3. **Planner Agent**: Analyzes the prompt and generates a high-level `Plan` (tech stack, features, required files).
4. **Architect Agent**: Reads the `Plan` and constructs a precise `TaskPlan` of atomic implementation steps.
5. **Coder Agent**: Executes the `TaskPlan`, outputting the exact file contents as a structured schema.
6. **Delivery**: The FastAPI backend returns a JSON payload containing the array of generated files.
7. **Frontend Rendering**: The React app parses the payload, rendering a dynamic File Explorer and Code Viewer.
8. **Export**: User downloads the entire in-memory structure as a `.zip` file generated natively via JSZip.

```text
+-------------------+        +---------------------------------------------------------+
|                   |        | Backend (FastAPI + LangGraph)                           |
|   User Browser    |        |                                                         |
|   (React/Vite)    |  POST  |  +---------+    +-----------+    +---------+    +-------+
|                   +------->|  | API     |--->| Planner   |--->|Architect|--->| Coder |
| - Enter prompt    |        |  | Router  |    | (Think)   |    | (Steps) |    | (Code)|
| - View Code       |        |  +---------+    +-----------+    +---------+    +-------+
| - Download .zip   |<-------+                 (Pydantic Structured Outputs)           |
|                   | JSON   |                                                         |
+-------------------+        +---------------------------------------------------------+
```

---

## 3. Full Tech Stack

### Backend
* **FastAPI**: Extremely fast, asynchronous Python web framework used for routing and request validation. 
* **LangGraph**: Orchestrates the multi-agent pipeline as a state machine. Allows for clear directional graphs (Planner -> Architect -> Coder) and conditional edges.
* **Pydantic**: Essential for the system's success. Enforces strict type-checking and schema outlines for the LLM outputs.
* **ChatGroq (LLama-3.3-70b-versatile)**: The underlying LLM engine. Chosen for its extreme inference speed, which is critical when chaining multiple complex LLM calls in a single HTTP request.

### Frontend
* **React + Vite**: For a snappy, hot-reloadable developer experience. 
* **TypeScript**: Provides compile-time safety and self-documenting code.
* **Tailwind CSS + shadcn/ui**: For a clean, modular, and professional user interface.
* **JSZip & FileSaver.js**: Handles the bundling and downloading of generated files seamlessly in the browser.

---

## 4. Complete Folder Structure

### `backend/`
The Python backend powering the orchestration.

* **`api.py`**: The FastAPI entry point. Defines the REST API, CORS middleware, and handles the request/response lifecycle.
* **`agent/graph.py`**: The brain of the LangGraph implementation. Wires together the nodes (`planner_agent`, `architect_agent`, `coder_agent`) and compiles the state graph.
* **`agent/prompts.py`**: Contains the system prompts and context injection logic for the LLMs.
* **`agent/states.py`**: Defines the Pydantic schemas (`Plan`, `TaskPlan`, `CoderOutput`) that force structured data through the pipeline graph.

### `frontend/src/`
The React app presenting the generated files.

* **`routes/` & `router.tsx`**: Defines the layout and views using file-based routing.
* **`components/ResultView.tsx`**: The main layout holding the File Explorer and Code Viewer components.
* **`components/FileExplorer.tsx`**: Recursively maps the flat `/` path strings into an interactive directory tree.
* **`components/CodeViewer.tsx`**: Syntactically highlights and presents the AI-generated code.
* **`lib/api.ts`**: The fetch abstract that calls the backend.
* **`lib/downloadZip.ts`**: Converts the JSON file array into a binary Blob and triggers a `.zip` download.

---

## 5. AI Pipeline Deep Dive

The most critical architectural decision in this project was **migrating away from Tool Calling and Raw JSON parsing**, choosing instead to use **LLM Structured Outputs**.

### 1. The Planner Agent (Brainstorming)
The Planner looks at the `user_prompt` and outputs a `Plan` schema. It focuses on the "What".
*(Temperature: 0.7 for slight creative variance)*

### 2. The Architect Agent (Decomposition)
The Architect takes the output of the Planner and outputs a `TaskPlan` schema, explicitly outlining atomic implementation steps for a developer. It focuses on the "How".
*(Temperature: 0.2 for logical grounding)*

### 3. The Coder Agent (Execution)
The Coder takes the `TaskPlan` and generates a `CoderOutput` schema, specifically mapping `files: List[FileItem]`.
*(Temperature: 0.0 for pure determinism, focusing strictly on valid syntax)*

### Why We Dropped Tool Calling & Raw Parsing
Originally, the Coder used OpenAI-style Tool Calling to "write" files to a virtual state. However, models struggled with sequence timing and often emitted `tool_use_failed` errors when generating large strings of code inside function arguments. Free-form JSON parsing was equally brittle. Switching to LangChain's `.with_structured_output(Schema, method="json_mode")` forces the LLM to reply natively with valid schema-compliant objects, completely bypassing tool execution complexities and drastically reducing failure rates.

---

## 6. API Documentation

### `POST /generate`
Generates a full codebase in memory.

**Request Body:**
```json
{
  "prompt": "Create a React todo app with local storage"
}
```

**Success Response (200 OK):**
```json
{
  "success": true,
  "files": [
    {
      "path": "src/App.jsx",
      "content": "import React from 'react';\n..."
    },
    {
      "path": "package.json",
      "content": "{\n  \"name\": \"todo-app\"\n}"
    }
  ]
}
```

**Error Response (422 Unprocessable Entity | 500 Internal Error):**
```json
{
  "success": false,
  "error": "Code generation failed",
  "details": "LLM failed to output valid JSON matching the CoderOutput Schema."
}
```

---

## 7. Frontend System and ZIP Download

Instead of writing files to disk on the backend and uploading them to S3/GCS or serving them via static routes, the backend returns everything as strings over the API.

**Why Javascript ZIP?**
By offloading the ZIP bundling to the user's browser using `JSZip`:
1. The backend becomes completely stateless.
2. We avoid massive I/O operations, disk locks, and cron jobs for cleaning up old projects.
3. The server requires zero database or file storage, drastically reducing deployment costs.
4. The user sees the files rendering dynamically on their screen via the `ResultView` and `FileExplorer` instantly.

---

## 8. Major Problems Faced During Development

### 1. `tool_use_failed` Errors
* **Problem**: Agents tasked with writing multiple files via function calling would often hallucinate arguments or timeout generating massive JSON strings inside tool schemas.
* **Solution**: Refactored the LangGraph pipeline to utilize strict Pydantic parsing natively (`with_structured_output`) instead of forcing the agent to invoke "write_file" functions.

### 2. Frontend/Backend Contract Mismatch & Malformed JSON
* **Problem**: Early iterations used standard prompt engineering ("Return a JSON object"). The LLMs frequently prefixed responses with \`\`\`json or broke standard formatting, crashing the frontend.
* **Solution**: Implementing exact Pydantic schemas over `json_mode` completely eradicated serialization errors, ensuring the React frontend always receives exactly `Array<{ path: string, content: string }>`.

---

## 9. Key Learnings

1. **Schema-Driven Orchestration > Prompt Engineering:** Giving an LLM an exact structural blueprint via code is exponentially more reliable than asking it nicely in English.
2. **Decomposition Enables Complexity:** A single prompt asking "Write a full app" fails catastrophically. Splitting it into Plan -> Architect -> Code works reliably.
3. **Stateless is Scalable:** Shipping JSON payload strings implies the backend can sit behind edge functions, scale infinitely, and ignore file-system permission nightmares. 

---

## 10. Setup Instructions

### Backend Setup
```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Environment Variables:**
Create a `.env` in the `backend/` directory (requires Groq API Key):
```
GROQ_API_KEY=gsk_your_groq_api_key
```

**Run the Server:**
/backend
```bash
uvicorn api:app --reload
# Server runs on http://localhost:8000
```

### Frontend Setup
```bash
cd frontend
bun install # or npm install
bun dev     # or npm run dev
```

---

## 11. Deployment Notes

* **Frontend**: Can be exported out to Vercel, Netlify, or Cloudflare Pages. Ensure `VITE_API_BASE` is configured properly in `.env.production` or platform variables.
* **Backend**: Deploys well to serverless container providers (Render, Railway, Heroku). 
* **CORS**: Ensure the `CORSMiddleware` in `api.py` reflects your production frontend domain to prevent unauthorized requests.

---

## 12. Future Improvements

* **Iterative Refinement (Validator Agent)**: Introduce a 4th agent that lints or type-checks the code output and feeds errors back into the Coder for auto-correction before returning to the user.
* **WebContainer Integration**: Run the generated code directly in the browser via WebContainers for an instant live-preview, entirely avoiding local installation.
* **Database Persistence / Auth**: Connect a database to allow users to save past generations and resume modifying projects.
* **External Integrations**: Add GitHub Export features for immediately pushing generated project files to new Repositories.

---

## 13. Conclusion

Agentic Web Builder serves as a practical, lightweight template for how deterministic multi-agent systems interact with modern web frameworks. By strictly typing API layers and relying intensely on schema-enforced LLM responses, it transforms volatile generative AI interactions into a dependable, production-ready developer tool.

