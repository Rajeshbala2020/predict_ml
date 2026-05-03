# Stock ML Service (FastAPI)

Production-ready Python ML service that trains a stock movement model, serves predictions, stores them in PostgreSQL, and triggers user notifications.

## Tech Stack

- Python 3.11
- FastAPI + Uvicorn
- pandas + numpy
- scikit-learn (RandomForestClassifier)
- yfinance
- SQLAlchemy + psycopg2
- APScheduler

## Project Structure

```text
ml-service/
  main.py
  train.py
  features.py
  db.py
  scheduler.py
  service.py
  docker-compose.yml
  Makefile
  .env.example
  requirements.txt
  README.md
  model.pkl (generated after training)
```

## Setup

1. Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start PostgreSQL (Docker):

```bash
docker compose up -d
```

4. Set database connection string:

```bash
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/ml_service"
```

## Database Setup

Create a PostgreSQL database and use these tables (auto-created on app start):

- `predictions`: `id`, `symbol`, `prediction`, `confidence`, `created_at`
- `watchlist`: `id`, `user_id`, `symbol`
- `notifications`: `id`, `user_id`, `symbol`, `message`, `is_read`, `created_at`

Optional seed example:

```sql
insert into watchlist (user_id, symbol) values
  ('user_1', 'AAPL'),
  ('user_2', 'AAPL'),
  ('user_1', 'MSFT');
```

If you use Docker compose, the database is already created as `ml_service` with `postgres/postgres`.

## Training

Train and persist `model.pkl`:

```bash
python train.py
```

The app auto-trains on first startup if `model.pkl` does not exist.

## Run Service

```bash
uvicorn main:app --reload --port 8000
```

### Full Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
export DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/ml_service"
python train.py
uvicorn main:app --reload --port 8000
```

## Makefile Commands

```bash
make setup        # create venv + install deps
make up           # start postgres container
make down         # stop postgres container
make logs         # follow postgres logs
make train        # train and save model.pkl
make test-predict # run direct predictor smoke test
make run          # run FastAPI service
make quickstart   # setup + up + train + run
```

## Scheduler / Cron Behavior

APScheduler runs every 5 minutes and does:

1. Read symbols from `watchlist`
2. Fetch market data
3. Build features
4. Predict direction (`UP`/`DOWN`)
5. Insert into `predictions`
6. Insert into `notifications` if:
   - confidence > 70, or
   - prediction direction changed from previous saved value

If watchlist is empty, it predicts the training default symbol (`AAPL`).

## API Usage

### `GET /`

Health/status endpoint.

### `POST /predict`

Request:

```json
{
  "symbol": "AAPL"
}
```

Response:

```json
{
  "symbol": "AAPL",
  "prediction": "UP",
  "confidence": 73.42
}
```

### `POST /predict-and-store`

Same request body as `/predict`. Runs prediction and persists result into `predictions`, while applying notification rules.

### `POST /watchlist`

Add a symbol to a user's watchlist.

```json
{
  "user_id": "user_1",
  "symbol": "AAPL"
}
```

### `GET /watchlist/{user_id}`

List watchlist symbols for a user.

### `DELETE /watchlist/{user_id}/{symbol}`

Remove a symbol from a user's watchlist.

### `GET /notifications/{user_id}?unread_only=false`

List notifications for a user (or unread only).

### `PATCH /notifications/{notification_id}/read`

Mark one notification as read.

### `GET /predictions?symbol=AAPL&limit=50`

List recent stored predictions with optional symbol filter.

## Notes

- Uses SQLAlchemy engine pooling (`pool_pre_ping`, `pool_size`, `max_overflow`)
- Includes error handling and logging in API and scheduler paths
- Prevents target leakage by constructing target from future close and splitting train/test chronologically
