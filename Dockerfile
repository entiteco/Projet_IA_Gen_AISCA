# Image de base Python légère
FROM python:3.10-slim

# Évite les fichiers temporaires .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dossier de travail dans le conteneur
WORKDIR /app

# Installation des outils système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du projet
COPY . .

# Port pour Streamlit
EXPOSE 8501

# Commande de lancement
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]