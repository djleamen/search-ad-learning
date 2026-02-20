# Search Ad Learning

Search UI that evolves into a word cloud and audience-segment view, 
backed by a Python machine-learning taxonomy service.

## Features

- Word cloud layout
- Full Python backend:
  - FastAPI API layer
  - Online-learning text classifier (hashing + SGD log-loss)
  - Event persistence in PostgreSQL (SQLAlchemy)
  - Taxonomy training corpus generation
  - Feedback + retraining endpoints

## Global model + per-user state

- Hidden global knowledge base (shared across all users):
  - `global_feedback_events`
  - `global_conversion_affinity`
  - `global_model_registry`
- Private per-user state (keyed by `user_id`, clearable):
  - `user_search_events`
  - `user_feedback_events`
  - `user_category_totals`
  - `user_tag_totals`
  - `user_embedding`
  - `user_conversion_affinity`

`POST /history/clear` only deletes `user_*` rows for the active user and does not touch global tables.

## Architecture

- Frontend: `index.html`, `styles.css`, `app.js`
- Backend package: `backend/`
  - `main.py` API endpoints
  - `taxonomy_data.py` taxonomy + seed corpus + tag expansion
  - `model_service.py` model training/inference/online updates
  - `store.py` persistent event and aggregate store

## Run frontend only

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## Run backend + frontend

### 1) Install backend dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) Set environment variables

```bash
export DATABASE_URL="postgresql+psycopg://<user>:<password>@<host>:5432/<db>"

# Optional: verify JWT bearer tokens signed with HS256.
# If unset, backend accepts X-User-Id header (used by frontend in local/dev).
export AUTH_JWT_SECRET="change-me"
```

### 3) Start backend API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

### 4) Start frontend

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

The frontend will call `http://127.0.0.1:8001/search` automatically.

## Azure deployment target

- Frontend: Azure Static Web Apps
- Backend: Azure Container Apps
- Database: Azure Database for PostgreSQL Flexible Server
- Secrets: Azure Key Vault
- Artifacts: Azure Blob Storage (recommended for model files)

For production, use Entra ID / B2C issued JWTs and avoid `X-User-Id` fallback.

### Backend container build (local)

```bash
docker build -f backend/Dockerfile -t search-ad-learning-backend:latest .
docker run --rm -p 8001:8001 \
  -e DATABASE_URL="postgresql+psycopg://<user>:<password>@<host>:5432/<db>" \
  search-ad-learning-backend:latest
```

## Backend endpoints

- `GET /health`
- `GET /taxonomy`
- `POST /search` with `{ "query": "..." }`
- `POST /feedback` with `{ "query": "...", "category": "...", "confidence": 1.0 }`
- `POST /retrain`
