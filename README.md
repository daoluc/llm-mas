# LLM-based Multi Agent System


## Running the FastAPI Server

To run the FastAPI server:

```bash
fastapi run agent_service.py
```

To update requirements:

```bash
pipreqs --force .
```

To run local agent api:

```bash
uvicorn agent_service:app --host 0.0.0.0 --port 8000 --workers 5
```
