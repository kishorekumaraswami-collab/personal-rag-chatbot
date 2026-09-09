FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY chunks_data.pkl .

# Expose port (Koyeb uses PORT env variable)
EXPOSE 8000

# Run the app
CMD ["python", "app.py"]
