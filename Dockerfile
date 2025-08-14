FROM python:3.11-slim

LABEL maintainer="Aniket Bhardwaj"
LABEL description="IFRS-16 LBO Engine: Reproducible research environment"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "pytest", "tests/"]

# For interactive use:
# docker build -t ifrs16-lbo .
# docker run -it -p 8501:8501 ifrs16-lbo streamlit run streamlit_app.py
