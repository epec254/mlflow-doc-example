from fastapi import FastAPI
import os
import uvicorn
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend to access backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/hello")
async def query():
    return "Hello, world!"


PORT = int(os.getenv("UVICORN_PORT", 8000))
HOST = os.getenv("UVICORN_HOST", "0.0.0.0")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
