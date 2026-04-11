#!/bin/bash
# Start FastAPI in background on port 8000
uvicorn api.routes:app --host 0.0.0.0 --port 8000 &

# Start Streamlit on the PORT given by Render
streamlit run ui/app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false \
  --server.runOnSave false