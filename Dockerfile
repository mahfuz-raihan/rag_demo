# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system-level dependencies
# - libgomp1: ABSOLUTELY REQUIRED for FAISS (the vector store)
# - build-essential: Necessary for some Python packages that need to compile C code
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file and install dependencies
# We do this before copying the whole app to leverage Docker's cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire project into the container
COPY . .

# 6. Chainlit runs on port 8000 by default
EXPOSE 8000

# 7. The Start Command
# --host 0.0.0.0: Mandatory for cloud services to access the container
# --port 8000: Matches the EXPOSE command
CMD ["chainlit", "run", "ui/app.py", "--host", "0.0.0.0", "--port", "8000"]