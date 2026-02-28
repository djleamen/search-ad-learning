# Search Ad Learning

**Visualization demo of how search engines determine ad topics and audience segments.**

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

# Search Ad Learning

Interactive search UI that evolves into a word cloud and audience-segment view, backed by a Python machine-learning taxonomy service.

This project demonstrates:
- Online learning with user feedback loops
- Per-user state isolation with shared global knowledge
- Full-stack ML system design (UI → API → model → persistence)
- Cloud-ready architecture (Azure target)

---

# System Overview

Search Ad Learning simulates how search intent signals evolve into audience segments and ad targeting signals.

High-level flow:

1. User submits a search query
2. Backend predicts category probabilities
3. UI updates segment view + word cloud
4. User provides feedback or conversion signals
5. Model updates incrementally (online learning)
6. State persists per-user while global signals accumulate

The system supports both:
- Backend ML mode (FastAPI + PostgreSQL)
- Local fallback learning (browser-only simulation)

---

# Architecture

## Frontend
- `index.html`
- `styles.css`
- `app.js`

Responsibilities:
- Query submission
- Visualization (word cloud + segment bars)
- Feedback collection
- Conversion click simulation
- Local fallback learning

The frontend dynamically resolves backend URLs and gracefully degrades if the API is offline.

## Backend (FastAPI)
Located in `backend/`:

- `main.py` – API routes
- `model_service.py` – Online learning classifier (hashing + SGD log-loss)
- `taxonomy_data.py` – Taxonomy definitions + seed corpus
- `store.py` – Persistence layer (SQLAlchemy)

Core API endpoints:
- `GET /health`
- `POST /search`
- `POST /feedback`
- `POST /retrain`
- `POST /history/clear`
- `POST /conversion/click`

---

# Data Model & State Isolation

## Global Tables
Shared learning across all users:
- `global_feedback_events`
- `global_conversion_affinity`
- `global_model_registry`

## Per-User Tables
Isolated by `user_id`:
- `user_search_events`
- `user_feedback_events`
- `user_category_totals`
- `user_tag_totals`
- `user_embedding`
- `user_conversion_affinity`

`POST /history/clear` only deletes `user_*` rows.

This separation simulates multi-tenant ad-learning systems.

---

# Model Design

The backend classifier uses:
- Hashing-based feature extraction
- SGD with log-loss (incremental updates)
- Online updates from feedback signals
- Confidence weighting for corrections

Learning sources:
- Initial taxonomy seed corpus
- Query text features
- Explicit user feedback
- Conversion click events

This allows the model to evolve without full retraining cycles.

---

# Cloud Deployment Target

Azure reference architecture:

- Frontend: Azure Static Web Apps
- Backend: Azure Container Apps
- Database: Azure Database for PostgreSQL Flexible Server
- Secrets: Azure Key Vault
- Model artifacts: Azure Blob Storage
- Auth: Entra ID / B2C JWT validation

Local development uses `X-User-Id` header fallback.

---

# CI/CD

GitHub Actions pipeline:
- Lint + test (planned expansion)
- Docker build
- Deploy to Azure Container Apps
- Static frontend deployment

Future improvements:
- Enforce test coverage thresholds
- Add model regression evaluation step
- Add security scanning (bandit / dependency audit)

---

# Testing & Evaluation Roadmap

Planned improvements:

## Software Testing
- Unit tests for model service
- API integration tests
- Persistence layer tests
- Failure case tests (invalid input, offline backend)

## Model Evaluation
- Offline validation dataset
- Accuracy / F1 tracking
- Confusion matrix export
- Drift tracking across retrain cycles

## RAG-Style Extension (Future)
- Retrieval quality metrics
- Signal weighting analysis
- Conversion lift tracking

---

# Security Considerations

- JWT verification support
- Per-user isolation
- No secrets in frontend
- Cloud secret management (Key Vault)
- Clear separation between local fallback and production mode

Future improvements:
- Rate limiting
- Feedback poisoning detection
- Structured audit logging

---

# Running Locally

## Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
export DATABASE_URL="postgresql+psycopg://<user>:<password>@<host>:5432/<db>"
uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

## Frontend

```bash
python3 -m http.server 8000
```

Open:

http://localhost:8000

---

# Why This Project Exists

This project explores:
- Online ML in interactive systems
- Feedback-driven model updates
- Multi-tenant ML state isolation
- Bridging frontend UX with ML pipelines

It is intentionally designed as a clean, non-technical entry point into applied ML system design.

---

# License

MIT
