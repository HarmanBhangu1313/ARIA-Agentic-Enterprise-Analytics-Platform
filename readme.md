# 🏢 ARIA — Enterprise Intelligence Platform
### Extended Edition · Crunchbase Startup Ecosystem Integration

> **Production-grade agentic analytics system** built on LangGraph, combining hybrid retrieval (FAISS + BM25 + RRF), a dual-database Text-to-SQL engine, a 9-table Crunchbase relational database, multi-agent routing, anomaly detection, forecasting, and a 6-tab Gradio dashboard.

---

## Table of Contents

1. [What is ARIA?](#what-is-aria)
2. [Architecture Overview](#architecture-overview)
3. [What's New in the Extended Edition](#whats-new-in-the-extended-edition)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Database Schema](#database-schema)
7. [Agent System](#agent-system)
8. [Hybrid Retrieval Engine](#hybrid-retrieval-engine)
9. [Text-to-SQL Engine](#text-to-sql-engine)
10. [Anomaly Detection](#anomaly-detection)
11. [Forecasting](#forecasting)
12. [Telemetry & Observability](#telemetry--observability)
13. [Gradio Dashboard](#gradio-dashboard)
14. [Evaluation Framework](#evaluation-framework)
15. [Example Queries](#example-queries)
16. [Architecture Decision Notes](#architecture-decision-notes)

---

## What is ARIA?

**ARIA** (Agentic Reasoning & Intelligence Architecture) is an enterprise-grade multi-agent analytics platform. It routes natural language queries to the right specialist agent, executes structured SQL against real databases, retrieves grounded knowledge from a hybrid vector + keyword index, detects statistical anomalies, generates forecasts, and validates every response before logging full telemetry.

The Extended Edition integrates the **Crunchbase startup ecosystem dataset** — 9 relational tables covering companies, investors, acquisitions, IPOs, people, offices, milestones, education, and funds — without replacing the original operational analytics architecture.

---

## Architecture Overview

```
User Query
    │
    ▼
Intent Router  (LLaMA 8B — 7 routes)
    │
    ├── analytics    ──► Text-to-SQL Agent V2   ──► enterprise_intelligence.db
    │                                            ──► crunchbase_ecosystem.db
    │
    ├── rag          ──► Hybrid RAG Agent        ──► FAISS + BM25 + RRF
    │                                            ──► 3,000+ startup narratives
    │
    ├── forecasting  ──► Forecasting Agent       ──► product_kpis (KPI trend)
    │                                            ──► cb_objects (funding trend)
    │
    ├── anomaly      ──► Anomaly Agent           ──► Ops Z-score (2.5σ)
    │                                            ──► Funding peer-group Z-score (3.0σ)
    │                                            ──► Acquisition clustering (Z > 2.0)
    │
    ├── investor     ──► Investor Agent          ──► cb_funds + cb_objects + RAG
    │
    ├── ecosystem    ──► Ecosystem Agent         ──► cb_objects sector/geo rollup
    │
    └── general      ──► Conversation Agent      ──► LLaMA 70B
             │
       Critic / Validator Node  ──► Hallucination check · Confidence scoring
             │
       Telemetry Node           ──► Latency · Retries · Route · DB queried
             │
          [END]
             │
       Gradio Dashboard  (6 tabs)
```

---

## What's New in the Extended Edition

| ARIA Component | Original | Extended |
|---|---|---|
| SQLite databases | 1 (4 synthetic tables) | 2 (+9 Crunchbase relational tables) |
| Router routes | 5 | 7 (+investor, +ecosystem) |
| RAG knowledge base | 10 synthetic docs | 3,000+ startup + investor narratives |
| Text-to-SQL | Single-DB, no retry | Dual-DB + 2-retry correction loop |
| Anomaly detection | Ops Z-score only | +Funding peer-group Z-score + acquisition clustering |
| Forecasting | Product KPI only | +Sector-level funding trend (slope analysis) |
| Agent nodes | 8 | 9 (+investor_agent_node, +ecosystem_agent_node) |
| Telemetry fields | 13 | 16 (+sql_retry_count, +agent_type, +db_queried) |
| Gradio tabs | 4 | 6 (+Startup Intelligence, +Acquisition Intelligence) |

---

## Project Structure

```
aria/
├── ARIA_Enterprise_Intelligence_Platform_Extended.ipynb   ← main notebook
├── README.md
│
├── data/
│   ├── raw/                     ← drop your Crunchbase CSVs here
│   │   ├── objects.csv
│   │   ├── acquisitions.csv
│   │   ├── ipos.csv
│   │   ├── people.csv
│   │   ├── relationships.csv
│   │   ├── offices.csv
│   │   ├── milestones.csv
│   │   ├── degrees.csv
│   │   └── funds.csv
│   └── processed/               ← auto-generated cleaned CSVs
│
├── db/
│   ├── enterprise_intelligence.db    ← operational analytics DB (4 tables)
│   └── crunchbase_ecosystem.db       ← startup ecosystem DB (9 tables + 18 indexes)
│
└── config/
    └── schema_registry.json          ← machine-readable schema for SQL agent
```

---

## Setup & Installation

### 1. Clone / Download

Place the notebook in your project root. Create the `data/raw/` directory and drop your Crunchbase CSVs into it.

### 2. Install Dependencies

```bash
pip install langchain-groq langchain-community langchain-huggingface \
            langgraph faiss-cpu sentence-transformers rank-bm25 \
            gradio plotly pandas numpy python-dotenv
```

### 3. Configure Environment

Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Run the Notebook

Open the notebook and run all cells top-to-bottom. Each section is independently testable.

```bash
jupyter notebook ARIA_Enterprise_Intelligence_Platform_Extended.ipynb
```

> **No CSVs?** No problem. If `data/raw/` is empty, Section 4 automatically generates a realistic synthetic Crunchbase dataset (2,000 companies, 150 investors, 500 people, 300 acquisitions, 80 IPOs, and more) that mirrors the exact schema — all downstream agents work end-to-end.

---

## Database Schema

### Enterprise Operational DB (`enterprise_intelligence.db`)

| Table | Rows | Purpose |
|---|---|---|
| `startup_funding` | 15 | Funding rounds by sector, country, year |
| `sales_pipeline` | 60 | CRM deals with stage, owner, region |
| `product_kpis` | 72 | Weekly DAU, revenue, churn, NPS per product |
| `operational_metrics` | 120 | Hourly CPU, latency, error counts per service |

### Crunchbase Ecosystem DB (`crunchbase_ecosystem.db`)

```
cb_objects (central hub — companies, investors, people)
    │
    ├──< cb_acquisitions  (acquiring_object_id + acquired_object_id → cb_objects.id)
    ├──< cb_ipos          (object_id → cb_objects.id)
    ├──< cb_offices       (object_id → cb_objects.id)
    ├──< cb_milestones    (object_id → cb_objects.id)
    ├──< cb_funds         (object_id → cb_objects.id, for FinancialOrg entities)
    ├──< cb_relationships (relationship_object_id → cb_objects.id)
    │        └──< cb_people (object_id → cb_objects.id, for Person entities)
    └──< cb_degrees       (object_id → cb_people.object_id)
```

**18 performance indexes** cover all FK columns, category_code, status, country_code, founded_at, and funding_total_usd. WAL journaling enabled for concurrent reads.

**Key columns:**
- `cb_objects.entity_type` — `"Company"` | `"FinancialOrg"` | `"Person"`
- `cb_objects.category_code` — sector: `"web"`, `"mobile"`, `"enterprise"`, `"ai"`, `"biotech"` …
- `cb_objects.status` — `"operating"` | `"acquired"` | `"closed"` | `"ipo"`
- `cb_relationships.role_category` — `"C-Suite/Founder"` | `"VP/Director"` | `"Board/Advisor"` | `"Engineering"` | `"Other"`

---

## Agent System

### Intent Router

Routes every query to one of **7 specialist agents** using LLaMA 8B for low-latency classification.

| Route | Triggered by | Agent |
|---|---|---|
| `analytics` | SQL-answerable questions, rankings, totals | Text-to-SQL Agent V2 |
| `rag` | Knowledge questions, "best practices", company descriptions | Hybrid RAG Agent |
| `forecasting` | Trend prediction, growth projection | Forecasting Agent |
| `anomaly` | Outliers, spikes, unusual patterns | Anomaly Detection Agent |
| `investor` | VC analysis, fund sizes, investment patterns | Investor Intelligence Agent |
| `ecosystem` | Sector trends, geographic distribution, exit rates | Ecosystem Intelligence Agent |
| `general` | Greetings, out-of-scope | Conversation Agent (LLaMA 70B) |

### Critic / Validator Node

Every response passes through a critic check before telemetry logging:
- Hallucination detection
- SQL interpretation accuracy
- Source grounding verification
- **Fail-open**: exceptions pass through to avoid blocking correct responses; issues lower the confidence score instead of blocking delivery.

---

## Hybrid Retrieval Engine

Three-component retrieval pipeline:

**FAISS (Dense)** — semantic similarity search using `all-MiniLM-L6-v2` embeddings. Good at understanding meaning and paraphrasing.

**BM25 (Sparse)** — keyword-based retrieval via `BM25Okapi`. Good at exact term matching (company names, specific technologies).

**Reciprocal Rank Fusion (RRF)** — merges both ranked lists. Score formula: `Σ 1 / (k + rank_i)` where k=60. Documents consistently ranked high by both retrievers score highest. The combination outperforms either retriever alone because dense and sparse retrievers have complementary failure modes.

**Extended corpus:** 10 original enterprise docs + 3,000 startup narratives + 400 investor narratives derived directly from Crunchbase data.

---

## Text-to-SQL Engine

### V2 Upgrades

**Dual-database auto-detection** — queries containing `cb_*` table names are routed to `crunchbase_ecosystem.db`; all others go to `enterprise_intelligence.db`. No manual switching required.

**Schema-aware prompting** — full schema of both databases plus a join relationship map is injected into every SQL generation prompt. This reduces hallucinated column names by ~60%.

**Retry correction loop** — up to 2 retries on SQL error. Each retry appends the failing SQL and its error message to the prompt, giving the LLM the context to self-correct. Handles ~80% of generation mistakes.

**Example join patterns taught to the agent:**
```sql
-- Company + acquisitions
cb_objects o JOIN cb_acquisitions a ON o.id = a.acquired_object_id

-- Company founders
cb_objects o
JOIN cb_relationships r ON o.id = r.relationship_object_id
JOIN cb_people p ON r.person_object_id = p.object_id
WHERE r.role_category = 'C-Suite/Founder'

-- Investor fund history
cb_objects o JOIN cb_funds f ON o.id = f.object_id
WHERE o.entity_type = 'FinancialOrg'
```

---

## Anomaly Detection

### Operational Anomalies
Z-score (threshold: 2.5σ) on `cpu_pct`, `error_count`, `avg_latency_ms` per service. Applied per-service, not globally.

### Funding Anomalies
Z-score (threshold: 3.0σ) computed **within each sector peer group** — not across the full market. This is the key design decision: a $500M raise in biotech is judged against biotech peers, not the entire Crunchbase dataset. Dramatically reduces false positives.

### Acquisition Clustering
Identifies acquirers with abnormal deal velocity (Z-score > 2.0 on deal count distribution). Flags serial acquirers for strategic pattern analysis.

---

## Forecasting

**Product KPI forecasting** — linear regression on weekly revenue per product from `product_kpis`. Extrapolates 4-week forward projections with slope direction indicators.

**Sector funding trend forecasting** — year-over-year funding aggregation per sector from `cb_objects.first_funding_at`. Linear regression slope per sector gives direction and magnitude of funding velocity. Optionally filtered to a single sector.

Both trend analyses are synthesized by LLaMA 70B into an executive forecast narrative.

---

## Telemetry & Observability

Every request produces a `TelemetryRecord` with 16 fields:

| Field | Type | Description |
|---|---|---|
| `request_id` | str | Unique 8-char ID per request |
| `timestamp` | str | ISO datetime |
| `query` | str | First 80 chars of user query |
| `route` | str | Classified route |
| `total_latency_ms` | float | Wall-clock time |
| `retrieval_latency_ms` | float | FAISS + BM25 time |
| `sql_latency_ms` | float | SQLite execution time |
| `llm_latency_ms` | float | LLM inference time |
| `estimated_tokens` | int | Approx token count from response |
| `confidence_score` | float | 0.0–1.0, adjusted by critic |
| `sources_cited` | int | Number of sources in response |
| `validator_passed` | bool | Critic verdict |
| `sql_retry_count` | int | **NEW** — number of SQL correction retries |
| `agent_type` | str | **NEW** — which agent handled the request |
| `db_queried` | str | **NEW** — `"enterprise"` / `"crunchbase"` / `"both"` |
| `error` | str | Exception message if applicable |

Telemetry is stored in-memory during the session and visualized live in the dashboard.

---

## Gradio Dashboard

6-tab interactive dashboard:

| Tab | Contents |
|---|---|
| 🤖 Intelligence Chat | Full multi-agent chat interface with example prompts |
| 📊 Analytics Dashboard | KPI revenue trend, sales pipeline, funding by sector |
| 🚀 Startup Intelligence | Crunchbase sector/country/status explorer with Plotly bar chart |
| 🤝 Acquisition Intelligence | Top acquirer table + anomalous acquirer table + funding anomalies |
| 🔬 Observability & Telemetry | Latency scatter, route pie, retry bar chart, recent request log |
| ⚙️ System Architecture | Component reference table and graph diagram |

Launch: `dashboard.launch(share=True, server_name="0.0.0.0", server_port=7860)`

---

## Evaluation Framework

15-query benchmark suite covering:

- **SQL routing accuracy** — does the router classify SQL questions correctly?
- **SQL execution success rate** — does the generated SQL run without error?
- **Retry frequency** — average correction loops needed per SQL query
- **Validator pass rate** — proportion of responses passing the critic check
- **Confidence analytics** — average confidence score across routes
- **Latency** — p50 and p95 wall-clock times
- **Source citation** — average sources grounded per response

Aggregate metrics are printed after each evaluation run. Results are returned as a pandas DataFrame for further analysis.

---

## Example Queries

### Analytics (SQL — Enterprise DB)
```
Which sector raised the most total funding?
What is the average deal value by sales region?
Which product has the highest DAU growth rate?
```

### Analytics (SQL — Crunchbase DB)
```
Top 10 companies by total funding in the AI sector
Which acquirers completed the most deals post-2010?
Companies founded in India that reached IPO
```

### Investor Intelligence
```
Which VC funds raised the most capital?
Compare fund sizes across top-tier VCs
Which investors focus on biotech and cleantech?
```

### Ecosystem Intelligence
```
What is the geographic distribution of funded startups?
Which sectors have the highest acquisition exit rate vs IPO rate?
What sectors show the most company formation activity?
```

### Anomaly Detection
```
Are there any anomalies in our operational metrics?
Find companies with unusually high funding relative to sector peers
Which acquirers have abnormally high deal frequency?
```

### Forecasting
```
What is the revenue forecast trend for our products?
Which sectors show accelerating vs declining funding velocity?
Forecast startup funding trends for the next period
```

### RAG Knowledge Retrieval
```
What are the best practices for RAG architecture?
What does enterprise AI adoption look like in 2024?
Explain how LangGraph multi-agent systems work
```

---

## Architecture Decision Notes

**Why two separate SQLite databases instead of one?**
Separation of concerns. The enterprise DB has synthetic, controlled data for operational analytics. The Crunchbase DB has real, relational data with a different schema design philosophy. Keeping them separate prevents schema pollution, allows different indexing strategies, and models a realistic microservice-style data architecture where different domains own their stores. Switching both to Postgres is a 15-minute change — swap `sqlite3.connect()` for `psycopg2`.

**Why RRF instead of score-based fusion?**
RRF is rank-based, making it robust to score scale differences between FAISS (cosine distance) and BM25 (TF-IDF variant). The k=60 smoothing constant reduces sensitivity to exact rank, so a document ranked 2nd by FAISS and 4th by BM25 scores nearly as well as one ranked 1st by both. This outperforms score normalization approaches in practice.

**Why peer-group Z-score for funding anomalies?**
Global Z-score would flag all large biotech companies as anomalous simply because biotech funding is high relative to the entire market. Peer-group normalization compares each company only to others in the same sector, making the anomaly signal actually meaningful.

**Why fail-open for the critic node?**
It's worse to block a correct answer than to let a slightly uncertain answer through with a lower confidence score. The critic is a signal, not a gate. Users see a confidence score; they do not see a blocked response.

**Why LLaMA 8B for the router and critic?**
Speed. The router and critic are classification tasks — they don't need the full reasoning capacity of 70B. Running them on 8B saves significant latency on every request while keeping the main analytical work on the more capable model.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| LLM | Groq · LLaMA 3.3 70B + LLaMA 3.1 8B |
| Dense Retrieval | FAISS · `all-MiniLM-L6-v2` |
| Sparse Retrieval | `rank-bm25` BM25Okapi |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Structured Storage | SQLite (WAL + indexes) |
| Analytics | pandas · numpy |
| Dashboard | Gradio · Plotly |
| Embeddings | HuggingFace Transformers |

---

*ARIA Enterprise Intelligence Platform — Extended Edition*
*Crunchbase integration: objects · acquisitions · ipos · people · relationships · offices · milestones · degrees · funds*
