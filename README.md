# SkillBridge - API de Recomendação de Cursos

Sistema de recomendação inteligente de cursos utilizando Machine Learning.

## 📋 Descrição

API REST desenvolvida em Flask que utiliza modelos de Machine Learning (Random Forest) para recomendar cursos personalizados baseados no perfil do usuário.

## 🎯 Funcionalidades

- Recomendação de cursos por Regressão (pontuação)
- Classificação de adequação do curso ao perfil
- API REST para integração com outros sistemas
- Filtros por carreira, experiência e preferências

## 🛠️ Tecnologias

- Python 3.10+
- Flask (API REST)
- Scikit-learn (ML)
- Pandas (manipulação de dados)
- Pickle (serialização de modelos)

## 🚀 Como Executar

### Instalar dependências:
```bash
pip install -r requirements.txt
```

### Executar API:
```bash
python app.py
```

A API estará disponível em: `http://localhost:5000`

## 📊 Endpoints

### Health Check
```
GET /health
```

### Recomendação de Cursos
```
POST /recomendar
```

**Body:**
```json
{
  "usuario": {
    "carreira_desejada": "Cientista de Dados",
    "nivel_experiencia": "Intermediário",
    "idade": 28,
    "anos_experiencia": 3,
    "escolaridade": "Superior Completo",
    "tempo_disponivel_semanal": 10
  },
  "cursos": [...],
  "quantidade": 10
}
```

## 📦 Arquivos do Projeto

- `app.py` - API Flask
- `skillbridge_ml_notebook.ipynb` - Notebook com análise e treinamento
- `modelo_regressao.pkl` - Modelo Random Forest Regressor
- `modelo_classificacao.pkl` - Modelo Random Forest Classifier
- `dataset_treino.csv` - Dataset de treinamento
- `features.json` - Configuração das features
- `requirements.txt` - Dependências

## 👥 Integrantes

- Bruno Vinicius Barbosa - RM566366
- João Pedro Bitencourt Goldoni - RM564339
- Marina Tamagnini Magalhães - RM561786

## 📅 Projeto

FIAP - Global Solution 2025
Disciplina: Artificial Intelligence & Chatbot
