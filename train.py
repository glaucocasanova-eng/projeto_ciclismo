# =============================================================
#  SMART CITY — PREDIÇÃO DE DEMANDA DE BICICLETAS
#  Script de Treinamento
# =============================================================
#
#  Estrutura do pipeline:
#
#  1. Carregar os dados
#  2. Traduzir colunas de texto para números
#  3. Criar variáveis derivadas (feature engineering)
#  4. Separar X (entradas) e y (alvo)
#  5. Dividir em treino e teste — train_test_split
#  6. Padronizar escalas — fit_transform no treino, transform no teste
#  7. Torneio com validação cruzada K-Fold (só no conjunto de treino)
#  8. Treinar o vencedor e avaliar no conjunto de teste (nunca visto)
#  9. Salvar modelo, scaler e metadados para o deploy
#
# =============================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────────────────────
#  TRADUÇÃO DAS COLUNAS DE TEXTO
#  O dataset armazena algumas informações como palavras.
#  Converter para números com ordem que faz sentido.
# ─────────────────────────────────────────────────────────────
TRADUCAO_ESTACAO  = {'spring': 1, 'summer': 2, 'autumn': 3, 'winter': 4}
TRADUCAO_CLIMA    = {'clear': 1, 'misty': 2, 'light rain': 3, 'heavy rain': 4}
TRADUCAO_SIM_NAO  = {'no': 0, 'yes': 1}
TRADUCAO_DIA      = {'sunday': 0, 'monday': 1, 'tuesday': 2, 'wednesday': 3,
                     'thursday': 4, 'friday': 5, 'saturday': 6}
TRADUCAO_MES      = {'january': 1, 'february': 2, 'march': 3, 'april': 4,
                     'may': 5, 'june': 6, 'july': 7, 'august': 8,
                     'september': 9, 'october': 10, 'november': 11, 'december': 12}


def traduzir_colunas(df):
    """Converte todas as colunas de texto para números."""
    df = df.copy()
    df['season']     = df['season'].map(TRADUCAO_ESTACAO)
    df['weathersit'] = df['weathersit'].map(TRADUCAO_CLIMA)
    df['holiday']    = df['holiday'].map(TRADUCAO_SIM_NAO)
    df['workingday'] = df['workingday'].map(TRADUCAO_SIM_NAO)
    df['weekday']    = df['weekday'].map(TRADUCAO_DIA)
    df['mnth']       = df['mnth'].map(TRADUCAO_MES)
    return df


# ─────────────────────────────────────────────────────────────
#  CRIAÇÃO DE VARIÁVEIS DERIVADAS (FEATURE ENGINEERING)
#  Cria variáveis que capturam padrões que o modelo não
#  conseguiria perceber diretamente nas colunas brutas.
# ─────────────────────────────────────────────────────────────
def criar_variaveis(df):
    """Adiciona variáveis derivadas com lógica de negócio."""
    df = df.copy()

    # Nível de rush — a demanda tem picos claros nos horários
    # de deslocamento. Não é linear: às 8h e 18h é muito maior
    # do que às 10h ou 21h.
    def nivel_rush(hr):
        if 7 <= hr <= 9 or 17 <= hr <= 19:
            return 3   # pico de rush
        elif 10 <= hr <= 16:
            return 2   # horário comercial
        elif 20 <= hr <= 22:
            return 1   # noite tranquila
        else:
            return 0   # madrugada

    df['periodo_dia']      = df['hr'].apply(nivel_rush)

    # Conforto térmico: sensação térmica × (1 − umidade)
    # Dias quentes e secos têm muito mais aluguéis
    df['conforto_termico'] = df['atemp'] * (1 - df['hum'])

    # Chuva em dia de folga afeta muito mais a demanda
    # do que chuva em dia útil (as pessoas precisam se deslocar)
    df['chuva_fds']        = (
        (df['weathersit'] >= 3) & (df['workingday'] == 0)
    ).astype(int)

    # Hora ao quadrado — o efeito do horário é curvo (em forma
    # de sino duplo), não linear. hr² ajuda o modelo a capturar isso.
    df['hr_sq']            = df['hr'] ** 2

    return df


# ─────────────────────────────────────────────────────────────
#  FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────
def treinar():

    # ── PASSO 1: Carregar os dados ────────────────────────────
    caminho = "dataset/Bike Sharing Dataset.csv"
    if not os.path.exists(caminho):
        print(f"Arquivo nao encontrado: {caminho}")
        return

    df = pd.read_csv(caminho)
    print(f"Dados carregados: {df.shape[0]:,} registros, {df.shape[1]} colunas")

    # ── PASSO 2 + 3: Preparar os dados ───────────────────────
    print("\nPreparando os dados...")
    df = traduzir_colunas(df)
    df = criar_variaveis(df)

    # Variáveis de entrada (X) — o que o modelo "enxerga"

    variaveis = [
        'hr', 'hr_sq',                                    # horário
        'mnth', 'weekday',                                # data
        'season', 'holiday', 'workingday', 'weathersit',  # contexto
        'temp', 'atemp', 'hum', 'windspeed',              # sensores
        'periodo_dia', 'conforto_termico', 'chuva_fds',   # derivadas
    ]
    alvo = 'cnt'

    X = df[variaveis]
    y = df[alvo]
    print(f"  {len(variaveis)} variáveis de entrada | {len(y):,} amostras")

    # ── PASSO 4: Dividir treino e teste ──────────────────────
    # Separar 25% dos dados para teste final — esses dados
    # o modelo nunca verá durante o treino.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"\n  Treino : {X_treino.shape[0]:,} amostras")
    print(f"  Teste  : {X_teste.shape[0]:,} amostras (nunca vistos no treino)")

    # ── PASSO 5: Padronizar as escalas ───────────────────────
    # IMPORTANTE: fit_transform APENAS no treino.

    padronizador = StandardScaler()
    X_treino_pad = padronizador.fit_transform(X_treino)  # aprende a escala
    X_teste_pad  = padronizador.transform(X_teste)       # só aplica

    # ── PASSO 6: Torneio com validação cruzada K-Fold ────────
    # Roda o torneio APENAS no conjunto de treino.
    # O conjunto de teste fica guardado para a avaliação final.
    # Cada modelo treina em 4 partes e testa na 5ª, 5 vezes.
    divisao_kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    candidatos = {
        "Regressão Linear (Ridge)": Ridge(),
        "Random Forest":            RandomForestRegressor(
                                        n_estimators=100,
                                        random_state=42,
                                        n_jobs=-1
                                    ),
    }

    print("\n" + "=" * 52)
    print("  TORNEIO — VALIDAÇÃO CRUZADA K-FOLD (no treino)")
    print("=" * 52)

    resultados = {}
    for nome, candidato in candidatos.items():
        notas_r2  = cross_val_score(candidato, X_treino_pad, y_treino,
                                    cv=divisao_kfold, scoring='r2')
        notas_mae = cross_val_score(candidato, X_treino_pad, y_treino,
                                    cv=divisao_kfold,
                                    scoring='neg_mean_absolute_error')
        resultados[nome] = notas_r2.mean()

        print(f"\n  {nome}:")
        print(f"    R² médio  : {notas_r2.mean():.4f}  "
              f"(variação: ±{notas_r2.std():.4f})")
        print(f"    MAE médio : {-notas_mae.mean():.0f} bikes/hora")

    # ── PASSO 7: Treinar o vencedor ───────────────────────────
    vencedor_nome = max(resultados, key=resultados.get)
    print(f"\n  Vencedor do torneio: {vencedor_nome}")

    # Retreinamos com TODO o conjunto de treino (não só 4/5)
    modelo_final = RandomForestRegressor(
        n_estimators=300,   # mais árvores = mais estável
        random_state=42,
        n_jobs=-1
    ) if vencedor_nome == "Random Forest" else Ridge()

    modelo_final.fit(X_treino_pad, y_treino)

    # ── PASSO 8: Avaliar no conjunto de teste (nunca visto) ───
    # Este é o número honesto — o modelo nunca viu esses dados.
    y_pred_teste = modelo_final.predict(X_teste_pad)

    r2_teste   = r2_score(y_teste, y_pred_teste)
    mae_teste  = mean_absolute_error(y_teste, y_pred_teste)
    rmse_teste = np.sqrt(mean_squared_error(y_teste, y_pred_teste))
    mape_teste = np.mean(np.abs((y_teste - y_pred_teste) / (y_teste + 1))) * 100

    print("\n" + "=" * 52)
    print(f"  DESEMPENHO NO TESTE (dados nunca vistos)")
    print("=" * 52)
    print(f"    R²   : {r2_teste:.4f}  — explica {r2_teste*100:.1f}% da variação")
    print(f"    MAE  : {mae_teste:.0f} bikes/hora de erro médio")
    print(f"    RMSE : {rmse_teste:.0f} bikes/hora")
    print(f"    MAPE : {mape_teste:.1f}% de erro percentual médio")

    # Quais variáveis foram mais decisivas?
    if hasattr(modelo_final, 'feature_importances_'):
        importancia = pd.Series(modelo_final.feature_importances_,
                                index=variaveis).sort_values(ascending=False)
        print(f"\n  Top 5 variáveis mais importantes:")
        for var, peso in importancia.head(5).items():
            barra = '█' * int(peso * 60)
            print(f"    {var:<22}: {barra}  ({peso:.3f})")

    # ── PASSO 9: Salvar tudo para o deploy ────────────────────

    os.makedirs('models', exist_ok=True)

    # Retreino final com 100% dos dados (treino + teste)
    # Agora que é validado o desempenho, usando tudo disponível.
    padronizador_final = StandardScaler()
    X_tudo_pad = padronizador_final.fit_transform(X)
    modelo_final.fit(X_tudo_pad, y)

    with open("models/cycling_demand_model.pkl", "wb") as f:
        pickle.dump(modelo_final, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(padronizador_final, f)
    with open("models/features.pkl", "wb") as f:
        pickle.dump(variaveis, f)
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump({
            'season':     TRADUCAO_ESTACAO,
            'weathersit': TRADUCAO_CLIMA,
            'holiday':    TRADUCAO_SIM_NAO,
            'workingday': TRADUCAO_SIM_NAO,
            'weekday':    TRADUCAO_DIA,
            'mnth':       TRADUCAO_MES,
        }, f)

    print("\n  Arquivos salvos em /models:")
    print("    cycling_demand_model.pkl  — modelo treinado")
    print("    scaler.pkl                — padronizador de escalas")
    print("    features.pkl              — lista de variáveis")
    print("    encoders.pkl              — dicionários de tradução")
    print("\nTreinamento concluido!\n")


if __name__ == "__main__":
    treinar()