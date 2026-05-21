# =============================================================
#  SMART CITY — DASHBOARD DE TELEMETRIA IoT
#  Interface de Predição de Demanda de Bicicletas
#  Pós-Graduação em IoT — IFSP | Glauco Casanova
# =============================================================

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from datetime import datetime

st.set_page_config(
    page_title="Smart City — Telemetria IoT",
    layout="wide",
    page_icon="🚲",
)

# ─────────────────────────────────────────────────────────────
#  CARREGAMENTO DO MODELO E ARTEFATOS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    arquivos = {
        "modelo":    "models/cycling_demand_model.pkl",
        "scaler":    "models/scaler.pkl",
        "variaveis": "models/features.pkl",
        "traducoes": "models/encoders.pkl",
    }
    carregados = {}
    for chave, caminho in arquivos.items():
        if not os.path.exists(caminho):
            return None
        with open(caminho, "rb") as f:
            carregados[chave] = pickle.load(f)
    return carregados

recursos = carregar_modelo()

# ─────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING — idêntica ao treinamento
# ─────────────────────────────────────────────────────────────
def calcular_nivel_rush(hr: int) -> int:
    if 7 <= hr <= 9 or 17 <= hr <= 19:
        return 3
    elif 10 <= hr <= 16:
        return 2
    elif 20 <= hr <= 22:
        return 1
    return 0


def montar_vetor(hr, mnth, weekday, season, holiday,
                 workingday, weathersit, temp, atemp, hum, windspeed):
    """
    Monta o vetor de entrada na ordem EXATA das variáveis do treino.
    Nota: 'yr' foi removido — o modelo foi treinado com ambos os anos
    (2011 e 2012) e generaliza por padrões de uso, não por ano histórico.
    """
    periodo_dia      = calcular_nivel_rush(hr)
    conforto_termico = atemp * (1 - hum)
    chuva_fds        = int(weathersit >= 3 and workingday == 0)
    hr_sq            = hr ** 2

    # Ordem exata: ['hr','hr_sq','mnth','weekday','season','holiday',
    #               'workingday','weathersit','temp','atemp','hum',
    #               'windspeed','periodo_dia','conforto_termico','chuva_fds']
    return np.array([[
        hr, hr_sq, mnth, weekday,
        season, holiday, workingday, weathersit,
        temp, atemp, hum, windspeed,
        periodo_dia, conforto_termico, chuva_fds,
    ]])


# ─────────────────────────────────────────────────────────────
#  CABEÇALHO
# ─────────────────────────────────────────────────────────────
st.title("🚲 Smart City — Predição de Demanda")
st.markdown(
    "Painel de telemetria IoT. Informe as condições atuais e o modelo "
    "estimará quantas bicicletas serão retiradas na próxima hora."
)
st.divider()

# ─────────────────────────────────────────────────────────────
#  BARRA LATERAL — ENTRADAS
# ─────────────────────────────────────────────────────────────
agora = datetime.now()

with st.sidebar:
    st.header("📡 Dados dos Sensores")

    # ── Data e Hora ──────────────────────────────────────────
    st.subheader("🕐 Quando?")

    hr = st.selectbox(
        "Hora atual (0–23h)",
        options=list(range(24)),
        index=agora.hour,
    )

    NOMES_MES = ["Janeiro","Fevereiro","Março","Abril","Maio","Junho",
                 "Julho","Agosto","Setembro","Outubro","Novembro","Dezembro"]
    mes_nome = st.selectbox("Mês", NOMES_MES, index=agora.month - 1)
    mnth = NOMES_MES.index(mes_nome) + 1

    NOMES_DIA = ["Domingo","Segunda","Terça","Quarta","Quinta","Sexta","Sábado"]
    dia_nome  = st.selectbox("Dia da semana", NOMES_DIA)
    weekday   = NOMES_DIA.index(dia_nome)

    st.divider()

    # ── Contexto ─────────────────────────────────────────────
    st.subheader("🏙️ Contexto")

    OPCOES_ESTACAO = ["Primavera", "Verão", "Outono", "Inverno"]
    season         = OPCOES_ESTACAO.index(
                         st.selectbox("Estação do ano", OPCOES_ESTACAO)
                     ) + 1

    OPCOES_CLIMA = [
        "Céu limpo / poucas nuvens",
        "Névoa / nublado",
        "Chuva leve / garoa",
        "Chuva forte / tempestade",
    ]
    weathersit = OPCOES_CLIMA.index(
                     st.selectbox("Condição climática", OPCOES_CLIMA)
                 ) + 1

    holiday    = int(st.toggle("É feriado?",  value=False))
    workingday = int(st.toggle("É dia útil?", value=True))

    st.divider()

    # ── Sensores Ambientais ───────────────────────────────────
    st.subheader("🌡️ Condições Ambientais")

    temp_c  = st.slider("Temperatura real (°C)",     0, 41, 22)
    atemp_c = st.slider("Sensação térmica (°C)",      0, 50, 22,
                        help="Temperatura sentida pelo corpo.")
    hum_p   = st.slider("Umidade relativa (%)",       0, 100, 55)
    wind_k  = st.slider("Velocidade do vento (km/h)", 0, 67, 12)

    # Normalização para a escala 0–1 do dataset
    temp_norm  = temp_c  / 41
    atemp_norm = atemp_c / 50
    hum_norm   = hum_p   / 100
    wind_norm  = wind_k  / 67

# ─────────────────────────────────────────────────────────────
#  BOTÃO DE PREDIÇÃO
# ─────────────────────────────────────────────────────────────
_, col_btn, _ = st.columns([2, 1, 2])
with col_btn:
    calcular = st.button("⚡ Calcular Demanda",
                         use_container_width=True,
                         type="primary")

if calcular:
    if recursos is None:
        st.error(
            "Modelo não encontrado em /models. "
            "Execute **python treinamento_final.py** primeiro."
        )
        st.stop()

    # Monta, padroniza e prediz
    entrada  = montar_vetor(
        hr, mnth, weekday, season, holiday, workingday, weathersit,
        temp_norm, atemp_norm, hum_norm, wind_norm
    )
    entrada_pad = recursos["scaler"].transform(entrada)
    previsao    = max(0, int(round(
        recursos["modelo"].predict(entrada_pad)[0]
    )))

    # ── Resultado ─────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Resultado da Previsão")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚲 Bikes estimadas",  f"{previsao}/hora")
    c2.metric("🌡️ Temperatura",      f"{temp_c} °C")
    c3.metric("💧 Umidade",          f"{hum_p}%")
    c4.metric("💨 Vento",            f"{wind_k} km/h")

    st.divider()

    # ── Análise contextual ────────────────────────────────────
    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown("**🔍 O que o modelo considerou**")

        nivel = calcular_nivel_rush(hr)
        label_periodo = {
            3: "🔴 Horário de pico (rush)",
            2: "🟡 Horário comercial",
            1: "🟢 Noite tranquila",
            0: "⚪ Madrugada",
        }
        st.write(f"- Período do dia: **{label_periodo[nivel]}**")

        conforto = atemp_norm * (1 - hum_norm)
        emoji_c  = "😊 Agradável" if conforto > 0.35 else "😓 Desconfortável"
        st.write(f"- Índice de conforto: **{conforto:.2f}** — {emoji_c}")

        if weathersit >= 3 and workingday == 0:
            st.warning("⚠️ Chuva em dia de folga — demanda tende a cair bastante.")
        if holiday:
            st.info("ℹ️ Feriado: padrão de uso diferente de um dia útil típico.")

    with col_dir:
        st.markdown("**📈 Leituras dos sensores (escala 0–1)**")
        grafico = pd.DataFrame({
            "Sensor": ["Temperatura", "Sensação térmica", "Umidade", "Vento"],
            "Nível":  [temp_norm, atemp_norm, hum_norm, wind_norm],
        })
        st.bar_chart(grafico.set_index("Sensor"), color="#1E88E5")

    # ── Classificação da demanda ──────────────────────────────
    st.divider()
    if previsao < 50:
        msg = f"🔵 Demanda **baixa** — {previsao} bikes/hora esperadas."
    elif previsao < 150:
        msg = f"🟡 Demanda **moderada** — {previsao} bikes/hora esperadas."
    elif previsao < 300:
        msg = f"🟠 Demanda **alta** — {previsao} bikes/hora esperadas."
    else:
        msg = f"🔴 Demanda **muito alta** — {previsao} bikes/hora. Verifique a frota!"

    st.markdown(f"### {msg}")

# ─────────────────────────────────────────────────────────────
#  RODAPÉ
# ─────────────────────────────────────────────────────────────
st.divider()
st.caption("Glauco Casanova | Pós-Graduação em IoT — IFSP")