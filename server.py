from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# Import the API from api/main.py
from api.main import app as api_app

app = FastAPI()

# Allow CORS (frontend calls backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API under /api
app.mount("/api", api_app)

# Serve the CSV folder if needed
app.mount("/public", StaticFiles(directory="public"), name="public")

# Serve index.html
@app.get("/")
def home():
    return FileResponse("index.html")

# Serve any local file
@app.get("/{path:path}")
def serve_file(path: str):
    if os.path.isfile(path):
        return FileResponse(path)
    return FileResponse("index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
