"""
SkillBridge - API de Recomendação de Cursos com IA
===================================================
Recomenda APENAS cursos com ID >= 10000

Integrantes:
- Bruno Vinicius Barbosa - RM566366
- João Pedro Bitencourt Goldoni - RM564339
- Marina Tamagnini Magalhães - RM561786
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Carregar modelos
try:
    with open('modelo_classificacao.pkl', 'rb') as f:
        modelo_classificacao = pickle.load(f)
    with open('modelo_regressao.pkl', 'rb') as f:
        modelo_regressao = pickle.load(f)
    with open('features.json', 'r') as f:
        features_config = json.load(f)
    logger.info("✓ Modelos carregados")
except Exception as e:
    logger.error(f"Erro: {e}")
    raise

# Mapeamentos
NIVEL_EXP_MAP = {'Junior': 1, 'Intermediário': 2, 'Senior': 3}
ESCOLARIDADE_MAP = {'Ensino Fundamental': 1, 'Ensino Médio': 2, 'Técnico': 3,
                    'Superior Incompleto': 4, 'Superior Completo': 5,
                    'Pós-graduação': 6, 'Mestrado': 7, 'Doutorado': 8}
NIVEL_CURSO_MAP = {'BASICO': 1, 'INTERMEDIARIO': 2, 'AVANCADO': 3}

# Cursos com ID >= 10000 por carreira
CARREIRAS_CURSOS = {
    'Desenvolvedor Full Stack': [10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10042, 10043],
    'Desenvolvedor Frontend': [10034, 10035, 10036, 10044, 10045, 10046, 10047, 10048, 10049, 10050, 10051, 10052, 10053],
    'Desenvolvedor Backend': [10037, 10038, 10054, 10055, 10056, 10057, 10058, 10059, 10060, 10061, 10062, 10063],
    'Desenvolvedor Mobile': [10064, 10065, 10066, 10067, 10068, 10069, 10070, 10071, 10072, 10073],
    'Cientista de Dados': [10074, 10075, 10076, 10077, 10078, 10079, 10080, 10081, 10082, 10083],
    'Engenheiro de Dados': [10084, 10085, 10086, 10087, 10088, 10089, 10090, 10091, 10092, 10093],
    'Analista de Dados': [10094, 10095, 10096, 10097, 10098, 10099, 10100, 10101, 10102, 10103],
    'DevOps Engineer': [10104, 10105, 10106, 10107, 10108, 10109, 10110, 10111, 10112, 10113],
    'Designer UX/UI': [10114, 10115, 10116, 10117, 10118, 10119, 10120, 10121, 10122, 10123],
    'Product Manager': [10124, 10125, 10126, 10127, 10128, 10129, 10130, 10131, 10132, 10133],
    'Arquiteto de Software': [10134, 10135, 10136, 10137, 10138, 10139, 10140, 10141, 10142, 10143],
    'Engenheiro de Machine Learning': [10144, 10145, 10146, 10147, 10148, 10149, 10150, 10151, 10152, 10153],
    'Especialista em Cloud': [10154, 10155, 10156, 10157, 10158, 10159, 10160, 10161, 10162, 10163],
    'Analista de Segurança': [10164, 10165, 10166, 10167, 10168, 10169, 10170, 10171, 10172, 10173],
    'QA/Tester': [10174, 10175, 10176, 10177, 10178, 10179, 10180, 10181, 10182, 10183]
}

def criar_features(perfil, curso):
    nivel_exp = NIVEL_EXP_MAP.get(perfil.get('nivel_experiencia', 'Junior'), 1)
    nivel_curso = NIVEL_CURSO_MAP.get(curso.get('nivel', 'BASICO'), 1)
    escolaridade = ESCOLARIDADE_MAP.get(perfil.get('escolaridade', 'Superior Completo'), 5)
    
    match_nivel = 1 if nivel_exp >= nivel_curso - 1 else 0
    tempo_semanal = perfil.get('tempo_disponivel_semanal', 5.0)
    carga_semanal = curso.get('carga_horaria', 10.0) / 4
    match_tempo = 1 if tempo_semanal >= carga_semanal else 0
    
    carreira = perfil.get('carreira_desejada', '')
    cursos_carreira = CARREIRAS_CURSOS.get(carreira, [])
    match_carreira = 1 if curso.get('id_curso') in cursos_carreira else 0
    
    return {
        'nivel_experiencia_num': nivel_exp, 'tempo_disponivel_semanal': tempo_semanal,
        'idade': perfil.get('idade', 25), 'anos_experiencia': perfil.get('anos_experiencia', 0),
        'escolaridade_num': escolaridade, 'nivel_curso_num': nivel_curso,
        'carga_horaria': curso.get('carga_horaria', 10.0),
        'avaliacao_media': curso.get('avaliacao_media', 4.0),
        'taxa_conclusao_media': curso.get('taxa_conclusao_media', 80.0),
        'popularidade_score': curso.get('popularidade_score', 50.0),
        'match_nivel': match_nivel, 'match_tempo': match_tempo,
        'match_carreira': match_carreira, 'progresso': 0.0
    }

def aplicar_regras_negocio(score_base, features, perfil, curso):
    """
    Camada de regras de negócio para ajustar score do modelo.
    Garante que recomendações façam sentido do ponto de vista de UX.
    """
    score = score_base
    nivel_usuario = NIVEL_EXP_MAP.get(perfil.get('nivel_experiencia', 'Junior'), 1)
    nivel_curso = NIVEL_CURSO_MAP.get(curso.get('nivel', 'BASICO'), 1)
    
    # PENALIZAÇÕES POR MISMATCH DE NÍVEL
    diff_nivel = nivel_curso - nivel_usuario
    
    if diff_nivel >= 2:  # Curso muito avançado (ex: Junior tentando Avançado)
        score *= 0.3
    elif diff_nivel == 1 and nivel_usuario == 1:  # Junior tentando Intermediário
        score *= 0.7
    elif diff_nivel <= -2:  # Curso muito básico (ex: Senior fazendo Básico)
        score *= 0.4
    elif diff_nivel == -1 and nivel_usuario == 3:  # Senior fazendo Intermediário
        score *= 0.7
    
    # BOOST PARA MATCH DE CARREIRA
    if features['match_carreira'] == 1:
        score *= 1.4  # +40% se alinhado com carreira
    else:
        score *= 0.6  # -40% se não alinhado
    
    # BOOST PARA MATCH DE TEMPO
    if features['match_tempo'] == 1:
        score *= 1.1  # +10% se tem tempo disponível
    
    # BOOST PARA CURSOS BEM AVALIADOS
    if features['avaliacao_media'] >= 4.7:
        score *= 1.1  # +10% para cursos excelentes
    
    return max(0, min(10, score))

def gerar_motivo(features, prob):
    m = []
    if features['match_carreira'] == 1: m.append("Alinhado com sua carreira")
    if features['match_nivel'] == 1: m.append("Nível adequado")
    if features['match_tempo'] == 1: m.append("Carga horária compatível")
    if features['avaliacao_media'] >= 4.5: m.append(f"Bem avaliado ({features['avaliacao_media']:.1f}/5)")
    if prob >= 0.7: m.append("Alta chance de conclusão")
    return ". ".join(m) + "." if m else "Recomendado para você"

@app.route('/', methods=['GET'])
def home():
    return jsonify({'api': 'SkillBridge', 'versao': '2.0', 'status': 'online'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'modelos': True})

@app.route('/recomendar', methods=['POST'])
def recomendar():
    """
    Endpoint de recomendação compatível com formato Java e frontend
    
    FORMATO JAVA:
    {
        "usuario_id": 1,
        "perfil": {
            "nivel_experiencia": "JUNIOR",
            "tempo_disponivel_semanal": 10.0,
            "idade": 25,
            "anos_experiencia_total": 2,
            "objetivo_carreira": "Cientista de Dados"
        },
        "cursos": [{...}],
        "top_n": 10
    }
    
    FORMATO FRONTEND (mantido para compatibilidade):
    {
        "usuario": {...},
        "cursos": [{...}],
        "quantidade": 10
    }
    """
    try:
        data = request.get_json()
        
        # DETECTAR FORMATO (Java ou Frontend)
        if 'perfil' in data and 'usuario_id' in data:
            # FORMATO JAVA
            logger.info("🔵 Requisição do Java detectada")
            
            perfil_java = data.get('perfil', {})
            cursos = data.get('cursos', [])
            quantidade = data.get('top_n', 10)
            
            # Mapear perfil Java para formato interno
            usuario = {
                'carreira_desejada': perfil_java.get('objetivo_carreira', ''),
                'nivel_experiencia': perfil_java.get('nivel_experiencia', 'Junior'),
                'idade': perfil_java.get('idade', 25),
                'anos_experiencia': perfil_java.get('anos_experiencia_total', 0),
                'escolaridade': 'Superior Completo',  # Padrão
                'tempo_disponivel_semanal': float(perfil_java.get('tempo_disponivel_semanal', 5.0))
            }
            
            # Mapear cursos Java (camelCase) para snake_case
            cursos_normalizados = []
            for c in cursos:
                curso_norm = {
                    'id_curso': c.get('id') or c.get('id_curso'),
                    'nome': c.get('nome'),
                    'descricao': c.get('descricao'),
                    'nivel': c.get('nivel'),
                    'carga_horaria': float(c.get('cargaHoraria') or c.get('carga_horaria') or 10),
                    'avaliacao_media': float(c.get('avaliacaoMedia') or c.get('avaliacao_media') or 4.0),
                    'taxa_conclusao_media': float(c.get('taxaConclusaoMedia') or c.get('taxa_conclusao_media') or 80),
                    'popularidade_score': float(c.get('popularidadeScore') or c.get('popularidade_score') or 50)
                }
                cursos_normalizados.append(curso_norm)
            
            cursos = cursos_normalizados
            
        else:
            # FORMATO FRONTEND
            logger.info("🟢 Requisição do Frontend detectada")
            usuario = data.get('usuario', {})
            cursos = data.get('cursos', [])
            quantidade = data.get('quantidade', 10)
        
        if not cursos:
            return jsonify({'sucesso': False, 'erro': 'Lista de cursos vazia'}), 400
        
        # ====== FILTRAR APENAS CURSOS COM ID >= 10000 ======
        cursos_validos = [c for c in cursos if c.get('id_curso', 0) >= 10000]
        
        logger.info(f"Total de cursos recebidos: {len(cursos)}")
        logger.info(f"Cursos com ID >= 10000: {len(cursos_validos)}")
        
        if not cursos_validos:
            return jsonify({
                'sucesso': False, 
                'erro': 'Nenhum curso com ID >= 10000 encontrado'
            }), 400
        
        recomendacoes = []
        for curso in cursos_validos:
            features = criar_features(usuario, curso)
            X_reg = pd.DataFrame([{k: features[k] for k in features_config['regressao']}])
            X_class = pd.DataFrame([{k: features[k] for k in features_config['classificacao']}])
            
            score_base = float(modelo_regressao.predict(X_reg)[0])
            score_base = max(0, min(10, score_base))
            
            # APLICAR REGRAS DE NEGÓCIO
            score_final = aplicar_regras_negocio(score_base, features, usuario, curso)
            
            prob = float(modelo_classificacao.predict_proba(X_class)[0][1])
            
            recomendacoes.append({
                'curso': {
                    'id_curso': curso.get('id_curso') or curso.get('id'),
                    'nome': curso.get('nome'),
                    'descricao': curso.get('descricao', ''),
                    'area': curso.get('area', 'Tecnologia'),
                    'nivel': curso.get('nivel'),
                    'carga_horaria': curso.get('carga_horaria'),
                    'avaliacao_media': curso.get('avaliacao_media'),
                    'taxa_conclusao_media': curso.get('taxa_conclusao_media', 80.0),
                    'popularidade_score': curso.get('popularidade_score', 50.0)
                },
                'score_relevancia': round(score_final, 2),
                'probabilidade_conclusao': round(prob, 2),
                'motivo': gerar_motivo(features, prob),
                'modelo_ia': 'RandomForest + Business Rules',
                'versao_modelo': '2.0'
            })
        
        recomendacoes.sort(key=lambda x: x['score_relevancia'], reverse=True)
        top = recomendacoes[:quantidade]
        
        # Adicionar rank
        for i, rec in enumerate(top, 1):
            rec['rank'] = i
        
        logger.info(f"✓ {len(top)} cursos recomendados")
        
        return jsonify({
            'usuario_id': data.get('usuario_id'),
            'timestamp': datetime.now().isoformat(),
            'total_recomendacoes': len(top),
            'recomendacoes': top
        })
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        return jsonify({'sucesso': False, 'erro': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SkillBridge API - Recomendação de Cursos")
    print("=" * 70)
    print("✓ Recomenda APENAS cursos com ID >= 10000")
    print("✓ http://localhost:5000")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)
