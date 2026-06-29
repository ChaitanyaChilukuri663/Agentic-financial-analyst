# Deploy

LedgerLens runs as a **Streamlit demo** and a **thin FastAPI service**. Both call Azure
OpenAI remotely, so the hosting itself is free — the only cost is a few cents of tokens.

## Run locally

```bash
pip install -e ".[demo]"
cp .env.example .env          # fill in Azure OpenAI endpoint + key

streamlit run ledgerlens/ui/app.py          # UI at http://localhost:8501
uvicorn ledgerlens.api.app:app --reload     # API at http://localhost:8000/docs
```

```bash
# API example
curl -s localhost:8000/answer -H "content-type: application/json" -d '{
  "question": "what was the percent change in net revenue from 2022 to 2023?",
  "context": "Net revenue was $5,829 million in 2023 and $5,735 million in 2022."
}'
```

## Free hosting (the portfolio demo — $0 infra)

### Streamlit Community Cloud (primary)

1. Push this repo to GitHub (done).
2. On [share.streamlit.io](https://share.streamlit.io): **New app** → pick the repo →
   main file `ledgerlens/ui/app.py`.
3. **Advanced settings → Secrets**: paste your `LEDGERLENS_*` values (same keys as
   `.env`). The app bridges `st.secrets` → environment at startup.
4. Deploy. (`requirements.txt` at the repo root is installed automatically.)

### Hugging Face Spaces (alternative)

Create a Space (Streamlit SDK), point it at this repo, and add the `LEDGERLENS_*` values as
Space **secrets**. A Docker Space can run the Streamlit UI *and* the FastAPI service together.

### FastAPI elsewhere (Render / Fly.io free tier)

Start command: `uvicorn ledgerlens.api.app:app --host 0.0.0.0 --port $PORT`. Set the
`LEDGERLENS_*` env vars in the host's dashboard. (Free tiers sleep when idle — fine for a demo.)

## Scale-up path on Azure (documented, **not enabled** — it would burn credits)

When traffic justifies it:

- **Containerize** (`Dockerfile`) and deploy to **Azure Container Apps** or **App Service**
  (with autoscale).
- Move secrets to **Azure Key Vault** + **managed identity** instead of env vars.
- Upgrade **Azure AI Search Free → Standard** for the hybrid index at scale (the retriever
  already sits behind an interface, so this is a config swap).

The live portfolio demo deliberately stays on a free host; Azure is the written
"when we need to scale" path only, so student credits go to tokens, not idle infra.
