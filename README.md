# 🚲 Smart City — Predição de Demanda de Bicicletas Compartilhadas

> Projeto de IoT com Machine Learning para previsão em tempo real da demanda de bicicletas urbanas.
> Pós-Graduação em IoT — IFSP | Glauco Casanova

---

## O que este projeto faz

Sensores IoT coletam dados ambientais e contextuais (temperatura, umidade, vento, hora do dia...).
Um modelo de Machine Learning processa essas leituras e estima quantas bicicletas serão alugadas
na próxima hora — permitindo que operadores reposicionem a frota antes que a demanda aconteça.

---

## Estrutura de arquivos

```
smart-city-bikes/
│
├── dataset/
│   └── Bike Sharing Dataset.csv     ← dados históricos (2011–2012)
│
├── models/                          ← gerado após o treinamento
│   ├── cycling_demand_model.pkl     ← modelo treinado
│   ├── scaler.pkl                   ← padronizador de escalas
│   ├── features.pkl                 ← lista de variáveis (ordem exata)
│   └── encoders.pkl                 ← dicionários de tradução de texto
│
├── treinamento_final.py             ← pipeline de treinamento
├── app.py                           ← dashboard Streamlit
├── requirements.txt
└── README.md
```

---

## Como executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/smart-city-bikes.git
cd smart-city-bikes
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Coloque o dataset na pasta correta
```
dataset/Bike Sharing Dataset.csv
```

### 4. Execute o treinamento
```bash
python treinamento_final.py
```

O script segue a mesma lógica das aulas:
- Traduz colunas de texto para números
- Cria variáveis derivadas (feature engineering)
- Separa treino e teste com `train_test_split`
- Aplica `StandardScaler` com `fit_transform` no treino e `transform` no teste
- Realiza torneio com validação cruzada K-Fold **apenas no conjunto de treino**
- Avalia o modelo final no conjunto de teste (dados nunca vistos)
- Salva todos os artefatos em `/models`

### 5. Inicie o dashboard
```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

---

## Dataset

**Fonte:** UCI Machine Learning Repository — Bike Sharing Dataset

| Propriedade      | Valor                           |
|------------------|---------------------------------|
| Registros        | 17.379 (granularidade horária)  |
| Período          | 2011–2012, Washington D.C.      |
| Variável alvo    | `cnt` — total de aluguéis/hora  |
| Valores nulos    | Nenhum                          |

### Por que excluímos `casual` e `registered`?

Essas duas colunas somam matematicamente o valor de `cnt` (nosso alvo).
Incluí-las seria trapacear: o modelo acertaria 100% no treino mas falharia
completamente em produção. Isso se chama **data leakage**.

### Por que removemos `yr` (ano)?

A coluna `yr` (0=2011, 1=2012) capturava o crescimento histórico da frota
entre os dois anos do dataset. Em produção, o modelo já foi treinado com
**ambos os anos juntos** — não faz sentido pedir ao operador que informe
"em qual ano histórico estamos". Removemos `yr` para que o modelo generalize
por padrões reais de uso (hora, clima, dia da semana), não por tendência
histórica que não se repetirá.

---

## Variáveis utilizadas

| Grupo       | Variável           | O que representa                               |
|-------------|-------------------|------------------------------------------------|
| Horário     | `hr`              | Hora do dia (0–23)                             |
| Horário     | `hr_sq`           | Hora ao quadrado — captura o efeito de pico    |
| Data        | `mnth`            | Mês (1–12)                                     |
| Data        | `weekday`         | Dia da semana (0=domingo … 6=sábado)           |
| Contexto    | `season`          | Estação (1=primavera … 4=inverno)              |
| Contexto    | `holiday`         | É feriado? (0 ou 1)                            |
| Contexto    | `workingday`      | É dia útil? (0 ou 1)                           |
| Contexto    | `weathersit`      | Condição climática (1=limpo … 4=tempestade)    |
| Sensor      | `temp`            | Temperatura real normalizada (0–1)             |
| Sensor      | `atemp`           | Sensação térmica normalizada (0–1)             |
| Sensor      | `hum`             | Umidade relativa normalizada (0–1)             |
| Sensor      | `windspeed`       | Velocidade do vento normalizada (0–1)          |
| Derivada    | `periodo_dia`     | Nível de rush (0=madrugada … 3=pico)           |
| Derivada    | `conforto_termico`| `atemp × (1 − hum)` — índice de bem-estar      |
| Derivada    | `chuva_fds`       | Chuva em dia de folga (0 ou 1)                 |

---

## Pipeline — alinhado com a metodologia das aulas

```
Dados brutos
    │
    ├─ Tradução categórica (text → número)
    ├─ Feature Engineering (variáveis derivadas)
    │
    ├─ train_test_split (75% treino / 25% teste)
    │       │
    │       ├─ StandardScaler.fit_transform(X_treino)
    │       ├─ StandardScaler.transform(X_teste)
    │       │
    │       ├─ Torneio K-Fold 5x (só no treino)
    │       │       ├─ Regressão Linear (Ridge) — baseline
    │       │       └─ Random Forest            — vencedor
    │       │
    │       └─ Avaliação no X_teste (nunca visto)
    │               R² / MAE / RMSE / MAPE
    │
    └─ Retreino final com 100% dos dados → .pkl
```

---

## Modelos e resultados

| Modelo            | R² (K-Fold treino) | R² (teste real) | MAE teste   |
|-------------------|--------------------|-----------------|-------------|
| Regressão Linear  | 0.564              | —               | —           |
| **Random Forest** | **0.857**          | **0.868**       | **~43 bikes/h** |

O R² no teste (0.868) é ligeiramente superior ao K-Fold (0.857) — sinal de
que o modelo generalizou bem e não houve overfitting.

---

## Tecnologias

| Biblioteca   | Versão  | Para que serve                    |
|--------------|---------|-----------------------------------|
| streamlit    | 1.35+   | Interface web / dashboard IoT     |
| scikit-learn | 1.4+    | Modelos, scaler, validação        |
| pandas       | 2.0+    | Manipulação dos dados             |
| numpy        | 1.26+   | Operações numéricas               |
| matplotlib   | 3.8+    | Visualizações                     |

---

## Autor

**Glauco Casanova**
Pos-Graduação em IoT — IFSP Catanduva

---

## Licença

Projeto acadêmico. Dataset original licenciado pela UCI Machine Learning Repository.