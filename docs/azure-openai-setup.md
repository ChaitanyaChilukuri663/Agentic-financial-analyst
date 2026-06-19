# Azure OpenAI setup (for LedgerLens)

LedgerLens uses **Azure OpenAI** with two model deployments:

| Purpose | Model | Default deployment name (in `config.py`) |
|---------|-------|------------------------------------------|
| Chat / planning | `gpt-4o-mini` | `gpt-4o-mini` |
| Embeddings (P3) | `text-embedding-3-small` | `text-embedding-3-small` |

> **Cost note:** Azure OpenAI is **pay-per-token with no idle/hourly charge** — creating the
> resource and deployments costs nothing; you're only billed for tokens used. `gpt-4o-mini`
> (~$0.15 / 1M input, $0.60 / 1M output) and `text-embedding-3-small` (~$0.02 / 1M) make the
> whole project cost a few dollars. The things that *do* burn credits while idle — Azure AI
> Search **Standard** tier, an always-on App Service — are **not** needed now (they're the P5
> scale-up path, documented but disabled).

## Portal steps (≈10 min)

1. **Create the resource.** [portal.azure.com](https://portal.azure.com) → **Create a resource**
   → search **Azure OpenAI** → **Create**.
   - Subscription: your student subscription.
   - Resource group: create one, e.g. `ledgerlens-rg`.
   - Region: pick one that offers both models (e.g. **East US 2** or **Sweden Central** — see
     model availability if a deployment is greyed out later).
   - Name: e.g. `ledgerlens-openai`.
   - Pricing tier: **Standard S0**.
   - Review + create.

2. **Deploy the models.** Open the resource → **Go to Azure AI Foundry portal** (or **Model
   deployments** → **Manage deployments**) → **Deploy model** → **Deploy base model**:
   - Deploy `gpt-4o-mini` — set the **deployment name** to `gpt-4o-mini`.
   - Deploy `text-embedding-3-small` — set the **deployment name** to `text-embedding-3-small`.
   - (If you choose different deployment names, update the `LEDGERLENS_AZURE_*_DEPLOYMENT`
     vars below to match.)

3. **Get endpoint + key.** Resource → **Keys and Endpoint** → copy the **Endpoint** and
   **KEY 1**.

4. **Fill `.env`** (copy from `.env.example`):

   ```bash
   LEDGERLENS_PROVIDER=azure
   LEDGERLENS_AZURE_OPENAI_ENDPOINT=https://ledgerlens-openai.openai.azure.com
   LEDGERLENS_AZURE_OPENAI_API_KEY=<KEY 1>
   LEDGERLENS_AZURE_OPENAI_API_VERSION=2024-10-21
   LEDGERLENS_AZURE_CHAT_DEPLOYMENT=gpt-4o-mini
   LEDGERLENS_AZURE_EMBED_DEPLOYMENT=text-embedding-3-small
   ```

5. **Smoke-test** with the determinism baseline on a tiny slice (a few cents):

   ```bash
   python -m ledgerlens.evaluation.finqa_qa data/finqa/dev.json 20
   ```

## CLI alternative

```bash
az group create -n ledgerlens-rg -l eastus2
az cognitiveservices account create -n ledgerlens-openai -g ledgerlens-rg \
  -l eastus2 --kind OpenAI --sku S0
az cognitiveservices account deployment create -n ledgerlens-openai -g ledgerlens-rg \
  --deployment-name gpt-4o-mini --model-name gpt-4o-mini --model-version <latest> \
  --model-format OpenAI --sku-capacity 1 --sku-name Standard
# repeat for text-embedding-3-small
az cognitiveservices account keys list -n ledgerlens-openai -g ledgerlens-rg
az cognitiveservices account show -n ledgerlens-openai -g ledgerlens-rg --query properties.endpoint
```

## Budget discipline

- Use **`gpt-4o-mini`**, not `gpt-4o`, everywhere except a small frontier-comparison slice (P4).
- Keep **Azure AI Search on the Free tier** (P3) — never Standard.
- Don't provision an always-on App Service; the portfolio demo runs on a free host (P5).
- Delete the resource group when you're done experimenting to zero out any residual cost.
