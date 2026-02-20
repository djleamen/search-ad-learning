# Search Ad Learning

Search UI that evolves into a word cloud and audience-segment view, backed by a Python machine-learning taxonomy service.

## Features

- Word cloud layout
- Full Python backend:
  - FastAPI API layer
  - Online-learning text classifier (hashing + SGD log-loss)
  - Event persistence in SQLite
  - Taxonomy training corpus generation
  - Feedback + retraining endpoints

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

### 2) Start backend API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

### 3) Start frontend

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000`.

The frontend will call `http://127.0.0.1:8001/search` automatically.

## Backend endpoints

- `GET /health`
- `GET /taxonomy`
- `POST /search` with `{ "query": "..." }`
- `POST /feedback` with `{ "query": "...", "category": "...", "confidence": 1.0 }`
- `POST /retrain`
