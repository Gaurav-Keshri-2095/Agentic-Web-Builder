# Testing Suite Changes Log

This document tracks all the modifications and newly created files introduced to set up the testing pipelines for both the backend (FastAPI/LangGraph) and frontend (React/Vite).

## Backend Testing (Python / Pytest)

### Modified Files
* **`backend/requirements.txt`**
  * Added `pytest`, `pytest-asyncio`, and `pytest-mock` to handle asynchronous unit testing and object patching.

### New Files
* **`backend/tests/conftest.py`**
  * Created a shared global test fixture exporting a `fastapi.testclient.TestClient` connected to the main app instance.
* **`backend/tests/test_api.py`**
  * Implemented HTTP endpoint tests for `/generate`.
  * Fully mocked the `generate_codebase` utility to verify our FastAPI `JSONResponse` objects successfully parse standard completions, 422 validations (missing prompts/states), and 500 exceptions.
* **`backend/tests/test_graph.py`**
  * Implemented isolated logic tests for the agent nodes: `planner_agent`, `architect_agent`, and `coder_agent`.
  * Patched the `ChatGroq` model constructors and `ainvoke` pipelines strictly using `AsyncMock` to completely avoid hitting actual Groq APIs and incurring token costs.

---

## Frontend Testing (React / Vitest)

### Modified Files
* **`frontend/package.json`**
  * Added `"test": "vitest run"` and `"test:watch": "vitest"` scripts mapped to npm.
* **`frontend/tsconfig.json`**
  * Added `"vitest.config.ts"` to the `"include"` array to pass TS validation formatting on the bundler level.

### New Files
* **`frontend/vitest.config.ts`**
  * Defined pure `vitest` logic indicating a `jsdom` testing environment, absolute path resolution mappings (`@/`), and pointing to the setup config.
* **`frontend/src/test/setup.ts`**
  * Established the `@testing-library/jest-dom` logic.
  * Added automatic DOM unmount/cleanup behavior between tests to prevent DOM bleeding.
  * Explicitly polyfilled `matchMedia` and `ResizeObserver` (since tools like `jsdom` do not native support modern Radix UI primitive events used in our UI).
* **`frontend/src/lib/utils.test.ts`**
  * Configured rigorous parsing tests for the `parseApiError()` function, securing that standard `AxiosError`, native Javascript `Error`, or untyped strings are seamlessly transformed into the `NormalizedError` shapes our UI expects.
* **`frontend/src/components/InputPanel.test.tsx`**
  * Implemented End-to-End browser DOM simulation using `Testing Library`.
  * Verified that clicking the generate button fires submit endpoints.
  * Verified that pressing the `<Enter>` key dispatches form generation natively.
  * Verified that pressing the `<Shift>` + `<Enter>` keystroke properly cancels submission to allow block formatting.
