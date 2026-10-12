"""Run the whole app locally: python serve_local.py, then open http://127.0.0.1:8000"""
from fastapi.staticfiles import StaticFiles
import uvicorn
from api.index import app

app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
