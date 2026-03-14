FROM python:3

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
COPY model/rental_prediction_model.pkl ./model/rental_prediction_model.pkl
COPY app.py ./

RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
#COPY . .

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "./app.py"]

