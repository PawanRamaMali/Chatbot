# Multi-stage build for Neural Chatbot
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r chatbot && useradd -r -g chatbot chatbot

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/share/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

# Copy application code
COPY src/ src/
COPY setup.py .
COPY README.md .

# Install the package
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data /app/plots /app/reports && \
    chown -R chatbot:chatbot /app

# Copy configuration files
COPY src/neural_chatbot/config/config.yaml /app/config.yaml
COPY src/neural_chatbot/data/intents.json /app/data/intents.json

# Switch to non-root user
USER chatbot

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command
CMD ["python", "-m", "neural_chatbot.api.app"]

# Production stage
FROM base as production

# Copy gunicorn configuration
COPY deployment/gunicorn.conf.py /app/gunicorn.conf.py

# Override CMD for production
CMD ["gunicorn", "--config", "gunicorn.conf.py", "neural_chatbot.api.app:create_app()"]

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Override CMD for development
CMD ["python", "-m", "neural_chatbot.api.app", "--debug"]