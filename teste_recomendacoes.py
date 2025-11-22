"""
Teste de Recomendações - SkillBridge
Valida se as recomendações fazem sentido
"""

import pandas as pd
import pickle
import json

print("="*80)
print("🧪 TESTE DE RECOMENDAÇÕES - SKILLBRIDGE")
print("="*80)

# Carregar modelos
print("\n📦 Carregando modelos...")
with open('modelo_classificacao.pkl', 'rb') as f:
    modelo_classificacao = pickle.load(f)
with open('modelo_regressao.pkl', 'rb') as f:
    modelo_regressao = pickle.load(f)
with open('features.json', 'r') as f:
    features_config = json.load(f)
print("✓ Modelos carregados")

# Carregar usuários
df_usuarios = pd.read_csv('usuarios.csv')
print(f"✓ {len(df_usuarios)} usuários carregados")

# Mapeamentos (igual app.py)
NIVEL_EXP_MAP = {'Junior': 1, 'Intermediário': 2, 'Senior': 3}
ESCOLARIDADE_MAP = {'Ensino Fundamental': 1, 'Ensino Médio': 2, 'Técnico': 3,
                    'Superior Incompleto': 4, 'Superior Completo': 5,
                    'Pós-graduação': 6, 'Mestrado': 7, 'Doutorado': 8}
NIVEL_CURSO_MAP = {'BASICO': 1, 'INTERMEDIARIO': 2, 'AVANCADO': 3}

CARREIRAS_CURSOS = {
    'Desenvolvedor Full Stack': [10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10042, 10043],
    'Desenvolvedor Frontend': [10034, 10035, 10036, 10044, 10045, 10046, 10047, 10048, 10049, 10050, 10051, 10052, 10053],
    'Desenvolvedor Backend': [10037, 10038, 10054, 10055, 10056, 10057, 10058, 10059, 10060, 10061, 10062, 10063],
    'Cientista de Dados': [10074, 10075, 10076, 10077, 10078, 10079, 10080, 10081, 10082, 10083],
    'Designer UX/UI': [10114, 10115, 10116, 10117, 10118, 10119, 10120, 10121, 10122, 10123],
}

# Criar cursos fictícios para teste
cursos_teste = [
    {'id_curso': 10074, 'nome': 'Python para Data Science', 'nivel': 'BASICO', 
     'carga_horaria': 40, 'avaliacao_media': 4.5, 'taxa_conclusao_media': 85, 'popularidade_score': 90},
    {'id_curso': 10075, 'nome': 'Machine Learning Fundamentos', 'nivel': 'INTERMEDIARIO',
     'carga_horaria': 60, 'avaliacao_media': 4.7, 'taxa_conclusao_media': 80, 'popularidade_score': 85},
    {'id_curso': 10076, 'nome': 'Deep Learning Avançado', 'nivel': 'AVANCADO',
     'carga_horaria': 80, 'avaliacao_media': 4.8, 'taxa_conclusao_media': 70, 'popularidade_score': 75},
    {'id_curso': 10114, 'nome': 'Fundamentos de UX Design', 'nivel': 'BASICO',
     'carga_horaria': 30, 'avaliacao_media': 4.6, 'taxa_conclusao_media': 88, 'popularidade_score': 82},
    {'id_curso': 10034, 'nome': 'HTML e CSS Básico', 'nivel': 'BASICO',
     'carga_horaria': 20, 'avaliacao_media': 4.4, 'taxa_conclusao_media': 90, 'popularidade_score': 95},
    {'id_curso': 10044, 'nome': 'React Avançado', 'nivel': 'AVANCADO',
     'carga_horaria': 50, 'avaliacao_media': 4.9, 'taxa_conclusao_media': 75, 'popularidade_score': 88},
]

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
        'nivel_experiencia_num': nivel_exp, 
        'tempo_disponivel_semanal': tempo_semanal,
        'idade': perfil.get('idade', 25), 
        'anos_experiencia': perfil.get('anos_experiencia', 0),
        'escolaridade_num': escolaridade, 
        'nivel_curso_num': nivel_curso,
        'carga_horaria': curso.get('carga_horaria', 10.0),
        'avaliacao_media': curso.get('avaliacao_media', 4.0),
        'taxa_conclusao_media': curso.get('taxa_conclusao_media', 80.0),
        'popularidade_score': curso.get('popularidade_score', 50.0),
        'match_nivel': match_nivel, 
        'match_tempo': match_tempo,
        'match_carreira': match_carreira, 
        'progresso': 0.0
    }

def aplicar_regras_negocio(score_base, features, perfil, curso):
    """Regras de negócio para ajustar score"""
    score = score_base
    nivel_usuario = NIVEL_EXP_MAP.get(perfil.get('nivel_experiencia', 'Junior'), 1)
    nivel_curso = NIVEL_CURSO_MAP.get(curso.get('nivel', 'BASICO'), 1)
    
    diff_nivel = nivel_curso - nivel_usuario
    
    if diff_nivel >= 2:
        score *= 0.3
    elif diff_nivel == 1 and nivel_usuario == 1:
        score *= 0.7
    elif diff_nivel <= -2:
        score *= 0.4
    elif diff_nivel == -1 and nivel_usuario == 3:
        score *= 0.7
    
    if features['match_carreira'] == 1:
        score *= 1.4
    else:
        score *= 0.6
    
    if features['match_tempo'] == 1:
        score *= 1.1
    
    if features['avaliacao_media'] >= 4.7:
        score *= 1.1
    
    return max(0, min(10, score))

def testar_perfil(perfil_usuario):
    print(f"\n{'='*80}")
    print(f"👤 PERFIL DE TESTE")
    print(f"{'='*80}")
    print(f"   Carreira: {perfil_usuario['carreira_desejada']}")
    print(f"   Nível: {perfil_usuario['nivel_experiencia']}")
    print(f"   Experiência: {perfil_usuario['anos_experiencia']} anos")
    print(f"   Tempo/semana: {perfil_usuario['tempo_disponivel_semanal']}h")
    print(f"   Idade: {perfil_usuario['idade']} anos")
    print(f"\n🎯 RECOMENDAÇÕES:\n")
    
    recomendacoes = []
    
    for curso in cursos_teste:
        features = criar_features(perfil_usuario, curso)
        
        X_class = pd.DataFrame([{k: features[k] for k in features_config['classificacao']}])
        X_reg = pd.DataFrame([{k: features[k] for k in features_config['regressao']}])
        
        prob = modelo_classificacao.predict_proba(X_class)[0][1]
        score_base = modelo_regressao.predict(X_reg)[0]
        score_base = max(0, min(10, score_base))
        
        # APLICAR REGRAS DE NEGÓCIO
        score = aplicar_regras_negocio(score_base, features, perfil_usuario, curso)
        
        recomendacoes.append({
            'nome': curso['nome'],
            'nivel': curso['nivel'],
            'carga': curso['carga_horaria'],
            'score': score,
            'prob': prob,
            'match_carreira': features['match_carreira'],
            'match_nivel': features['match_nivel'],
            'match_tempo': features['match_tempo']
        })
    
    recomendacoes.sort(key=lambda x: x['score'], reverse=True)
    
    for i, rec in enumerate(recomendacoes, 1):
        print(f"{i}. {rec['nome']}")
        print(f"   📊 Score: {rec['score']:.2f}/10")
        print(f"   ✅ Conclusão: {rec['prob']*100:.0f}%")
        print(f"   🎯 Match Carreira: {'SIM' if rec['match_carreira'] else 'NÃO'}")
        print(f"   📚 Nível: {rec['nivel']} | Carga: {rec['carga']}h")
        if rec['match_nivel']: print(f"   ✓ Nível adequado")
        if rec['match_tempo']: print(f"   ✓ Tempo compatível")
        print()

# ========== PERFIS DE TESTE ==========

# Teste 1: Cientista de Dados Júnior
perfil1 = {
    'carreira_desejada': 'Cientista de Dados',
    'nivel_experiencia': 'Junior',
    'idade': 24,
    'anos_experiencia': 1,
    'escolaridade': 'Superior Completo',
    'tempo_disponivel_semanal': 15
}

testar_perfil(perfil1)

# Teste 2: Designer UX/UI Intermediário
perfil2 = {
    'carreira_desejada': 'Designer UX/UI',
    'nivel_experiencia': 'Intermediário',
    'idade': 28,
    'anos_experiencia': 3,
    'escolaridade': 'Pós-graduação',
    'tempo_disponivel_semanal': 10
}

testar_perfil(perfil2)

# Teste 3: Frontend Senior com pouco tempo
perfil3 = {
    'carreira_desejada': 'Desenvolvedor Frontend',
    'nivel_experiencia': 'Senior',
    'idade': 35,
    'anos_experiencia': 8,
    'escolaridade': 'Superior Completo',
    'tempo_disponivel_semanal': 5
}

testar_perfil(perfil3)

print("\n" + "="*80)
print("✅ TESTES CONCLUÍDOS")
print("="*80)
