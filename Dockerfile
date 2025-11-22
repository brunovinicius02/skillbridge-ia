FROM python:3.11-slim

WORKDIR /app

# Copiar arquivos
COPY requirements.txt .
COPY app_ml.py app.py
COPY modelo_classificacao.pkl .
COPY modelo_regressao.pkl .
COPY features.json .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor porta
EXPOSE 7860

# Comando de inicialização
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
