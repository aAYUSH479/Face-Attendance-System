# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask default port
EXPOSE 8080

# Command to run the app
CMD ["python", "app.py"]
