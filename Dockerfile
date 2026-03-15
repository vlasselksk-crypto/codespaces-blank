# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY . /app

# Default env
ENV SLOTHAC_API_KEY=test-key

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
