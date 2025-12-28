# -----------------------------
# Base image
# -----------------------------
FROM python:3.9-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements & install
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy entire project
# -----------------------------
COPY . .

# -----------------------------
# Create uploads directory
# -----------------------------
RUN mkdir -p uploads

# -----------------------------
# Expose ports
# -----------------------------
EXPOSE 8000
EXPOSE 8501

# -----------------------------
# Start FastAPI & Streamlit
# -----------------------------
CMD ["bash", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
