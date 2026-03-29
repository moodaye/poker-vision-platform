"""run.py – start the Poker Vision Snipper API with uvicorn."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5003, reload=True)
