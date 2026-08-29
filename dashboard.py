import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
import requests
import re
import time
from requests.exceptions import Timeout, ConnectionError

HEADERS = {} 

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


modelo = xgb.XGBClassifier()
modelo.load_model('nba_wp_model_v2.json')

app = dash.Dash(__name__)


def data_br_para_espn(data_str):
    """Converte 'MM/DD/YYYY' para 'YYYYMMDD'"""
    mes, dia, ano = data_str.strip().split('/')
    return f"{ano}{mes.zfill(2)}{dia.zfill(2)}"


def converter_clock_espn(clock_str, period):

    try:
        if not clock_str or ':' not in clock_str:
            return None
        partes = clock_str.split(':')
        minutos = int(partes[0])
        segundos = float(partes[1])
        seg_no_periodo = minutos * 60 + segundos

        if period is None or period < 1:
            return None

        if period <= 4:
            return int(seg_no_periodo + (4 - period) * 720)
        else:
            overtime_index = period - 4
            return int(seg_no_periodo + (overtime_index - 1) * 300)
    except (ValueError, IndexError, TypeError):
        return None


def fetch_with_retry(fn, retries=3, wait=3):
    ultimo_erro = None
    for attempt in range(retries):
        try:
            return fn()
        except (Timeout, ConnectionError) as e:
            ultimo_erro = e
            if attempt < retries - 1:
                time.sleep(wait)
    raise ultimo_erro


def figura_vazia(mensagem):
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="#0B0E14",
        paper_bgcolor="#0B0E14",
        font=dict(color="white"),
        annotations=[dict(
            text=mensagem,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#EAA104")
        )]
    )
    return fig


#layout
app.layout = html.Div(style={'backgroundColor': "#0B0E14", 'color': 'white', 'padding': '20px', 'fontFamily': 'sans-serif'}, children=[
    html.H1("NBA Live Win Probability HUB", style={'textAlign': 'center', 'color': "#00D2FF", 'marginBottom': '40px'}),

    html.Div(style={
        'backgroundColor': '#1a1d24',
        'padding': '20px',
        'borderRadius': '10px',
        'marginBottom': '30px',
        'border': '1px solid #333',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '30px'
    }, children=[
        html.Img(
            src=app.get_asset_url('logonba.png'),
            style={'width': '150px', 'borderRadius': '5px'}
        ),
        html.Div([
            html.H3("How it works", style={'color': '#00D2FF', 'marginTop': '0'}),
            html.P([
                "This model uses ", html.B("XGBoost Machine Learning"),
                " to calculate the real-time victory probability. It analyzes ",
                html.Span("Time Remaining, Score Margin, and Ball Possession", style={'color': '#00D2FF'}),
                " based on historical NBA data from the last two seasons to predict the winner at any given moment. ",
                "First you choose the day you want, click in the ", html.B("Search"), " buttom and then select the game you want to see. ",
                html.Br(),
                html.Br(),
                "⚠️ Data provided by the public ESPN API"
            ], style={'fontSize': '15px', 'lineHeight': '1.6'})
        ])
    ]),

    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'paddingLeft': '20px', 'gap': '15px'}, children=[

        html.Div(style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-end', 'gap': '10px'}, children=[
            html.Div([
                html.Label('Game Day (MM/DD/YYYY)', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '5px', 'fontSize': '14px'}),
                dcc.Input(id='input-data', value='05/10/2024', type='text', style={
                    'backgroundColor': "#FFFFFF",
                    'color': '#1F1B2E',
                    'border': '1px solid #00D2FF',
                    'padding': '0 10px',
                    'borderRadius': '5px',
                    'height': '40px',
                    'width': '180px',
                    'outline': 'none',
                    'boxSizing': 'border-box'
                })
            ]),
            html.Button('Search', id='btn-busca', n_clicks=0, style={
                'cursor': 'pointer',
                'backgroundColor': '#1F1B2E',
                'color': "#EAA104",
                'border': '2px solid #EAA104',
                'borderRadius': '5px',
                'fontWeight': "bold",
                'height': '40px',
                'width': '100px',
                'boxSizing': 'border-box'
            })
        ]),

        html.Div([
            html.Label("Select the Game", style={"fontWeight": 'bold', 'display': 'block', 'marginBottom': '5px', 'fontSize': '14px'}),
            dcc.Dropdown(
                id='dropdown-jogos',
                placeholder="Select a matchup...",
                style={
                    'backgroundColor': '#1F1B2E',
                    'color': '#1F1B2E',
                    'border': '1px solid #00D2FF',
                    'borderRadius': '5px',
                    'width': '290px',
                    'height': '40px'
                }
            )
        ]),

        html.Div(id='status-msg', style={'color': '#EAA104', 'fontSize': '13px', 'minHeight': '18px'})
    ]),

    dcc.Loading(id="loading", type="graph", children=[
        dcc.Graph(id='graph-live', style={'height': '70vh', 'marginTop': '20px'})
    ])
])


#callback da API da ESPN
@app.callback(
    [Output('dropdown-jogos', 'options'),
     Output('status-msg', 'children')],
    Input('btn-busca', 'n_clicks'),
    State('input-data', 'value')
)
def update_dropdown(n_clicks, data_alvo):
    if n_clicks == 0:
        return [], ""

    try:
        data_espn = data_br_para_espn(data_alvo)
    except Exception:
        return [], "⚠️ Data inválida. Use o formato MM/DD/YYYY."

    time.sleep(0.5) 

    try:
        resposta = fetch_with_retry(lambda: requests.get(
            SCOREBOARD_URL,
            params={"dates": data_espn},
            headers=HEADERS,
            timeout=15
        ))
        resposta.raise_for_status()
        dados = resposta.json()
    except (Timeout, ConnectionError):
        return [], "⚠️ Timeout ao contatar a API da ESPN. Tente novamente em instantes."
    except requests.exceptions.HTTPError as e:
        if resposta.status_code == 403:
            return [], "⚠️ 403 Forbidden — a ESPN bloqueou temporariamente essa requisição (possível rate limit). Aguarde alguns segundos e tente de novo, ou desative o modo debug do Dash."
        return [], f"⚠️ Erro HTTP ao buscar jogos: {e}"
    except Exception as e:
        return [], f"⚠️ Erro inesperado ao buscar jogos: {e}"

    eventos = dados.get('events', [])
    if not eventos:
        return [], "Nenhum jogo encontrado nessa data."

    options = []
    for evento in eventos:
        game_id = evento['id']
        competidores = evento['competitions'][0]['competitors']
        home = next(c for c in competidores if c['homeAway'] == 'home')
        away = next(c for c in competidores if c['homeAway'] == 'away')
        label = f"{away['team']['abbreviation']} @ {home['team']['abbreviation']} ({game_id})"
        options.append({'label': label, 'value': game_id})

    return options, ""



@app.callback(
    [Output('graph-live', 'figure'),
     Output('status-msg', 'children', allow_duplicate=True)],
    Input('dropdown-jogos', 'value'),
    [State('dropdown-jogos', 'options'), State('input-data', 'value')],
    prevent_initial_call=True
)
def update_graph(game_id, options, value):
    if not game_id or not options:
        return figura_vazia("Selecione um jogo"), ""

    game_label = next((opt['label'] for opt in options if opt['value'] == game_id), game_id)
    game_name = game_label.split(' (')[0]

    try:
        resposta = fetch_with_retry(lambda: requests.get(
            SUMMARY_URL,
            params={"event": game_id, "region": "us", "lang": "en", "contentorigin": "espn"},
            headers=HEADERS,
            timeout=15
        ))
        resposta.raise_for_status()
        dados = resposta.json()
    except (Timeout, ConnectionError):
        return figura_vazia("Timeout na API da ESPN"), "⚠️ Timeout ao buscar o play-by-play. Tente novamente."
    except Exception as e:
        return figura_vazia("Erro ao carregar dados do jogo"), f"⚠️ Erro inesperado: {e}"

    plays = dados.get('plays', [])
    if not plays:
        return figura_vazia("Sem dados de play-by-play para este jogo"), "Essa partida não tem play-by-play detalhado disponível na ESPN."

    # descobre o id do time da casa para calcular a posse
    boxscore_teams = dados.get('boxscore', {}).get('teams', [])
    home_team_id = next((t['team']['id'] for t in boxscore_teams if t.get('homeAway') == 'home'), None)

    df = pd.DataFrame(plays)

    if 'sequenceNumber' in df.columns:
        df['sequenceNumber'] = pd.to_numeric(df['sequenceNumber'], errors='coerce')
        df = df.sort_values('sequenceNumber').reset_index(drop=True)

    df['clock_display'] = df['clock'].apply(lambda c: c.get('displayValue') if isinstance(c, dict) else None)
    df['period_number'] = df['period'].apply(lambda p: p.get('number') if isinstance(p, dict) else None)
    df['seconds_remaining'] = df.apply(lambda x: converter_clock_espn(x['clock_display'], x['period_number']), axis=1)

    #remover tempos não válidos 
    df = df.dropna(subset=['seconds_remaining'])
    if df.empty:
        return figura_vazia("Não foi possível interpretar o tempo de jogo das jogadas"), ""
    df['seconds_remaining'] = df['seconds_remaining'].astype(int)

    df['scoreHome'] = pd.to_numeric(df['homeScore'], errors='coerce').ffill().fillna(0)
    df['scoreAway'] = pd.to_numeric(df['awayScore'], errors='coerce').ffill().fillna(0)
    df['score_margin_home'] = df['scoreHome'] - df['scoreAway']

    df['team_id'] = df['team'].apply(lambda t: t.get('id') if isinstance(t, dict) else None)
    df['home_has_possession'] = (df['team_id'] == home_team_id).astype(int)

   #unicidade do tempo
    df = df.drop_duplicates(subset='seconds_remaining', keep='last').reset_index(drop=True)

    X = df[['seconds_remaining', 'score_margin_home', 'home_has_possession']]

    try:
        df['win_prob'] = modelo.predict_proba(X)[:, 1] * 100
    except Exception as e:
        return figura_vazia("Erro ao rodar o modelo"), f"⚠️ Erro no modelo XGBoost: {e}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['seconds_remaining'], y=df['win_prob'],
        mode='lines', line=dict(color="#00D2FF", width=4, shape='spline'),
        fill='tozeroy', fillcolor="rgba(0, 210, 255, 0.08)",
        name='Win Probability', showlegend=False
    ))

    #linha tracejada em 50%
    fig.add_hline(
        y=50, line_dash="dash", line_color="rgba(255, 255, 255, 0.4)", line_width=1.5
    )

    #quem venceu o jogo de fato,
    margem_final = df['score_margin_home'].iloc[-1]
    home_venceu = margem_final > 0

    if home_venceu:
        idx_min = df['win_prob'].idxmin()
        valor_min = df.loc[idx_min, 'win_prob']
        fig.add_trace(go.Scatter(
            x=[df.loc[idx_min, 'seconds_remaining']],
            y=[valor_min],
            mode='markers',
            marker=dict(color="#A855F7", size=13, line=dict(color="white", width=1.5)),
            name=f"Menor chance de vitória (venceu): {valor_min:.1f}%",
            hovertemplate=f"Menor probabilidade antes da virada: {valor_min:.1f}%<extra></extra>"
        ))
    else:
        idx_max = df['win_prob'].idxmax()
        valor_max = df.loc[idx_max, 'win_prob']
        fig.add_trace(go.Scatter(
            x=[df.loc[idx_max, 'seconds_remaining']],
            y=[valor_max],
            mode='markers',
            marker=dict(color="#F97316", size=13, line=dict(color="white", width=1.5)),
            name=f"Maior chance de vitória (perdeu): {valor_max:.1f}%",
            hovertemplate=f"Maior probabilidade antes de perder: {valor_max:.1f}%<extra></extra>"
        ))

    fig.update_layout(
        title=f"Live Win Probability for {game_name} - {value}",
        template="plotly_dark",
        plot_bgcolor="#0B0E14",
        paper_bgcolor="#0B0E14",
        xaxis=dict(title="Seconds Remaining", autorange="reversed", gridcolor="#1F1B2E"),
        yaxis=dict(title="Win Probability (%)", range=[0, 100], dtick=10, gridcolor="#1F1B2E")
    )

    return fig, ""


if __name__ == '__main__':
    app.run(debug=True)