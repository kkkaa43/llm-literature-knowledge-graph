# Deployment

Live demo: [https://llm-literature-kg.streamlit.app/](https://llm-literature-kg.streamlit.app/)

The public demo should default to the mock provider so visitors can run the app without API keys.

## Streamlit Community Cloud

Use these settings:

```text
Repository: kkkaa43/llm-literature-knowledge-graph
Branch: main
Main file path: app/streamlit_app.py
Python version: 3.11
```

No secrets are required for mock mode. If you enable real LLM providers, add keys only through the platform's private secrets manager:

```text
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
```

## Render

This repo includes `render.yaml`. In Render:

1. Create a new Blueprint.
2. Select this GitHub repository.
3. Keep the free web service settings.
4. Add `OPENROUTER_API_KEY` or `GEMINI_API_KEY` only as private environment variables if needed.

The start command is:

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```
