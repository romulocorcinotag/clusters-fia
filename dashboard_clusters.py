"""
Dashboard de Clusters FIA - TAG Investimentos
==============================================
Visualizacao interativa dos clusters de fundos FIA.

5 paginas:
  1. Visao Geral - Metricas resumo + treemap + stats
  2. Mapa de Clusters - UMAP 2D scatter (ou PCA fallback)
  3. Perfil do Cluster - Radar + sector breakdown + lista
  4. Analise de Fundo - Busca individual + peers
  5. Comparacao - Side-by-side de fundos
"""

import os
import base64
import sqlite3
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ============================================================
# PALETA TAG (Dark Theme)
# ============================================================
TAG_VERMELHO = "#630D24"
TAG_VERMELHO_LIGHT = "#8B1A3A"
TAG_VERMELHO_DARK = "#3D0816"
TAG_OFFWHITE = "#E6E4DB"
TAG_LARANJA = "#FF8853"
TAG_BRANCO = "#FFFFFF"
TAG_BG_DARK = "#1E0C14"
TAG_BG_CARD = "#2D1722"
TEXT_COLOR = TAG_OFFWHITE
TEXT_MUTED = "#9A9590"
CHART_GRID = "rgba(230,228,219,0.08)"

# Alturas padronizadas para charts
CHART_HEIGHT_FULL = 550    # Scatter UMAP, retorno acumulado principal
CHART_HEIGHT_MEDIUM = 400  # Radar, pie, treemap, retorno comparativo
CHART_HEIGHT_SMALL = 300   # Bars de comparacao

# Paleta de clusters: cores maximamente distintas no fundo escuro
# Cada cor eh unica em matiz, saturacao e luminosidade para facil identificacao
TAG_CHART_COLORS = [
    "#FF8853",  #  1 Laranja quente
    "#3B82F6",  #  2 Azul royal
    "#22C55E",  #  3 Verde vivo
    "#FACC15",  #  4 Amarelo ouro
    "#EF4444",  #  5 Vermelho vivo
    "#06B6D4",  #  6 Ciano
    "#A855F7",  #  7 Roxo
    "#EC4899",  #  8 Pink/magenta
    "#14B8A6",  #  9 Teal
    "#F97316",  # 10 Tangerina
    "#8B5CF6",  # 11 Violeta
    "#10B981",  # 12 Esmeralda
    "#F43F5E",  # 13 Rosa forte
    "#0EA5E9",  # 14 Sky blue
    "#84CC16",  # 15 Lima
]

# Features usadas no clustering
FEATURE_COLS = [
    # Retorno (18)
    'ann_return', 'ann_volatility', 'sharpe_approx', 'beta_ibov',
    'alpha_ibov', 'tracking_error', 'downside_vol', 'max_drawdown',
    'corr_ibov', 'autocorr_1d',
    'beta_smll', 'corr_smll', 'beta_idiv', 'corr_idiv',
    'up_capture', 'down_capture', 'smll_affinity', 'idiv_affinity',
    # Holdings / Carteira (6)
    'hhi_portfolio', 'pct_smallcap', 'pct_largecap',
    'pct_commodities', 'pct_utilities', 'turnover',
]

FEATURE_LABELS = {
    'ann_return': 'Retorno Anual',
    'ann_volatility': 'Volatilidade',
    'sharpe_approx': 'Sharpe',
    'beta_ibov': 'Beta IBOV',
    'alpha_ibov': 'Alpha',
    'tracking_error': 'Tracking Error',
    'downside_vol': 'Vol Downside',
    'max_drawdown': 'Max Drawdown',
    'corr_ibov': 'Corr IBOV',
    'autocorr_1d': 'Autocorrelacao',
    'beta_smll': 'Beta SMLL',
    'corr_smll': 'Corr SMLL',
    'beta_idiv': 'Beta IDIV',
    'corr_idiv': 'Corr IDIV',
    'up_capture': 'Up Capture',
    'down_capture': 'Down Capture',
    'smll_affinity': 'Afinidade SMLL',
    'idiv_affinity': 'Afinidade IDIV',
    # Holdings / Carteira
    'hhi_portfolio': 'Concentracao (HHI)',
    'pct_smallcap': '% Small Cap',
    'pct_largecap': '% Large Cap',
    'pct_commodities': '% Commodities',
    'pct_utilities': '% Utilities',
    'turnover': 'Turnover',
    'pct_dividend_stocks': '% Dividendos',
    'n_positions': 'Num. Posicoes',
}

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Clusters FIA - TAG Investimentos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, '_dados_cvm', 'cotas_cache.db')
HIST_PATH = os.path.join(APP_DIR, 'historico_fundos.pkl')

# Logo TAG
LOGO_SIDEBAR_PATH = os.path.join(APP_DIR, '..', 'luz_amarela', 'logo_sidebar.png')
if not os.path.exists(LOGO_SIDEBAR_PATH):
    LOGO_SIDEBAR_PATH = os.path.join(APP_DIR, '..', 'tag_logo.png')


def _get_logo_b64():
    """Retorna logo TAG em base64 para exibir no sidebar."""
    if os.path.exists(LOGO_SIDEBAR_PATH):
        with open(LOGO_SIDEBAR_PATH, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    return None


# ============================================================
# CSS
# ============================================================
def inject_css():
    _css = (
        '<style>'
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');"
        'html, body, [class*="css"] { font-family: "Inter", sans-serif; }'
        f'h1 {{ color: {TAG_OFFWHITE} !important; font-weight: 600 !important;'
        f' border-bottom: 2px solid {TAG_LARANJA}40; padding-bottom: 12px !important; }}'
        f'h2, h3 {{ color: {TAG_OFFWHITE} !important; font-weight: 500 !important; }}'
        '.stMainBlockContainer { max-width: 1400px; padding-top: 0.5rem !important; }'
        '#MainMenu {visibility: hidden;}'
        'footer {visibility: hidden;}'
        'header {visibility: hidden;}'
        f'section[data-testid="stSidebar"] {{'
        f' background: linear-gradient(180deg, {TAG_VERMELHO_DARK} 0%, {TAG_BG_DARK} 100%) !important; }}'
        f'section[data-testid="stSidebar"] .stRadio label {{'
        f' color: {TAG_OFFWHITE} !important; font-size: 1.05rem !important; }}'
        f'.metric-card {{ background: {TAG_BG_CARD}; border-radius: 12px; padding: 20px;'
        f' border-left: 4px solid {TAG_LARANJA}; margin-bottom: 10px; }}'
        f'.metric-card h3 {{ color: {TEXT_MUTED} !important; font-size: 0.85rem !important;'
        ' margin: 0 !important; text-transform: uppercase; letter-spacing: 0.05em; }'
        f'.metric-card .value {{ color: {TAG_OFFWHITE}; font-size: 2rem; font-weight: 700; }}'
        '.metric-card .delta { font-size: 0.85rem; margin-top: 4px; font-weight: 600; }'
        f'.metric-card .subtitle {{ color: {TEXT_MUTED}; font-size: 0.75rem; margin-top: 2px; }}'
        'section[data-testid="stSidebar"] .stRadio > div { gap: 2px; }'
        'section[data-testid="stSidebar"] .stRadio label {'
        ' padding: 8px 12px !important; border-radius: 6px !important; transition: background 0.2s; }'
        'section[data-testid="stSidebar"] .stRadio label:hover {'
        ' background: rgba(255,136,83,0.1) !important; }'
        '</style>'
    )
    st.markdown(_css, unsafe_allow_html=True)


# ============================================================
# CHART HELPERS
# ============================================================
def _chart_layout(title="", height=500, showlegend=True):
    """Layout padrao Plotly TAG."""
    return dict(
        title=dict(text=title, font=dict(color=TAG_OFFWHITE, size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=TAG_OFFWHITE, family='Inter'),
        height=height,
        showlegend=showlegend,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=TAG_OFFWHITE, size=11),
        ),
        xaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID),
        yaxis=dict(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID),
        margin=dict(l=40, r=20, t=50, b=40),
    )


def metric_card(label, value, delta=None, delta_color=None, subtitle=None):
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "–"
        d_color = delta_color or ("#22C55E" if delta > 0 else "#EF4444" if delta < 0 else TEXT_MUTED)
        delta_html = f'<div class="delta" style="color:{d_color};">{arrow} {abs(delta):.2f}</div>'
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="color:{TEXT_MUTED};font-size:0.75rem;margin-top:2px;">{subtitle}</div>'
    _mc_html = (
        f'<div class="metric-card">'
        f'<h3>{label}</h3>'
        f'<div class="value">{value}</div>'
        f'{delta_html}'
        f'{subtitle_html}'
        f'</div>'
    )
    st.markdown(_mc_html, unsafe_allow_html=True)


def _sparkline_svg(series, width=80, height=24, color=TAG_LARANJA):
    """Generate inline SVG sparkline from a pandas Series."""
    if series is None or len(series) < 2:
        return ""
    vals = series[-60:] if isinstance(series, np.ndarray) else series.tail(60).values
    mn, mx = float(vals.min()), float(vals.max())
    rng = mx - mn if mx != mn else 1
    points = []
    for i, v in enumerate(vals):
        x = i * width / (len(vals) - 1)
        y = height - (float(v) - mn) / rng * height
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return (f'<svg width="{width}" height="{height}" style="display:inline-block;vertical-align:middle;">'
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>')


def _build_radar_chart(traces, height=CHART_HEIGHT_MEDIUM):
    """Build a radar chart with one or more traces.

    Each trace in `traces` is a dict with keys:
      - values: list of floats (0-1 scale)
      - labels: list of str (axis labels)
      - name: str
      - color: str (hex color)
      - fill: bool (default True)
      - dash: str (default None, e.g. 'dot')
    """
    fig = go.Figure()
    for t in traces:
        vals = list(t['values']) + [t['values'][0]]
        theta = list(t['labels']) + [t['labels'][0]]
        # Convert hex color to rgba with 0.19 alpha for fill
        _hc = t['color'].lstrip('#')
        _r, _g, _b = int(_hc[:2], 16), int(_hc[2:4], 16), int(_hc[4:6], 16)
        fill_color = f'rgba({_r},{_g},{_b},0.19)' if t.get('fill', True) else 'rgba(0,0,0,0)'
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=theta,
            fill='toself' if t.get('fill', True) else 'none',
            fillcolor=fill_color,
            line=dict(color=t['color'], width=2, dash=t.get('dash')),
            name=t['name'],
        ))
    fig.update_layout(
        **_chart_layout("", height=height),
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=CHART_GRID,
                          tickfont=dict(size=9, color=TEXT_MUTED)),
            angularaxis=dict(gridcolor=CHART_GRID,
                           tickfont=dict(size=10, color=TAG_OFFWHITE)),
        ),
    )
    return fig


def _compute_radar_values(data_row, radar_features, all_data):
    """Compute z-score normalized values (0-1) for radar chart.

    data_row: Series or dict with feature values
    radar_features: list of feature column names
    all_data: DataFrame of all funds (for global mean/std)
    Returns: list of floats (0-1)
    """
    global_means = all_data[radar_features].mean()
    global_stds = all_data[radar_features].std().replace(0, 1)
    vals = []
    for f in radar_features:
        v = data_row.get(f, np.nan) if isinstance(data_row, dict) else data_row.get(f, np.nan)
        if pd.notna(v):
            z = (v - global_means[f]) / global_stds[f]
            vals.append(float(np.clip((z + 2) / 4, 0, 1)))
        else:
            vals.append(0.5)
    return vals


RADAR_FEATURES = ['corr_ibov', 'beta_ibov', 'ann_volatility', 'tracking_error',
                  'sharpe_approx', 'up_capture', 'down_capture',
                  'smll_affinity', 'idiv_affinity', 'pct_smallcap',
                  'hhi_portfolio', 'max_drawdown']


def _get_cluster_description(label, metadata_df):
    """Return auto_description for a cluster label from cluster_metadata table."""
    if metadata_df is None or metadata_df.empty:
        return 'Grupo de fundos com perfil quantitativo semelhante.'
    row = metadata_df[metadata_df['cluster_label'] == label]
    if row.empty:
        return 'Grupo de fundos com perfil quantitativo semelhante.'
    r = row.iloc[0]
    return r.get('auto_description', 'Grupo de fundos com perfil quantitativo semelhante.')


def _get_tactical_note(label, metadata_df):
    """Return tactical_note for a cluster label from cluster_metadata table."""
    if metadata_df is None or metadata_df.empty:
        return ''
    row = metadata_df[metadata_df['cluster_label'] == label]
    if row.empty:
        return ''
    r = row.iloc[0]
    return r.get('tactical_note', '')


# ============================================================
# DATA LOADING  (SQLite local  OU  parquet em data/)
# ============================================================
DATA_DIR = os.path.join(APP_DIR, 'data')
_USE_PARQUET = not os.path.exists(DB_PATH)


def _read_table(name, conn=None, query=None):
    """Le tabela do SQLite ou do parquet equivalente em data/."""
    if _USE_PARQUET:
        path = os.path.join(DATA_DIR, f'{name}.parquet')
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_parquet(path)
    try:
        q = query or f'SELECT * FROM {name}'
        return pd.read_sql_query(q, conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_data():
    """Carrega todos os dados — SQLite local ou parquet (cloud)."""
    conn = None
    if not _USE_PARQUET:
        if not os.path.exists(DB_PATH):
            st.error(f"Database nao encontrada: {DB_PATH}")
            st.stop()
        conn = sqlite3.connect(DB_PATH)

    # Cluster features (drop cluster_id/label to avoid suffix conflicts on merge)
    features_df = _read_table('cluster_features', conn)
    features_df = features_df.drop(columns=['cluster_id', 'cluster_label'], errors='ignore')

    # Cluster results (authoritative source for cluster_id, cluster_label, nearest_peers)
    results_df = _read_table('cluster_results', conn)

    # Sector breakdown
    sector_df = _read_table('sector_breakdown', conn)

    # Holdings features
    holdings_df = _read_table('holdings_features', conn)

    # Cluster metadata (taxonomy descriptions, tactical notes)
    metadata_df = _read_table('cluster_metadata', conn)

    # UMAP/PCA coordinates para visualizacao
    viz_df = _read_table('viz_coords', conn)

    # Cotas diarias: NAO carregar tudo aqui — sera carregado sob demanda
    cotas_df = pd.DataFrame()  # placeholder

    # Benchmark
    if _USE_PARQUET:
        ibov_df = _read_table('benchmark_daily', conn)
        if not ibov_df.empty and 'benchmark' in ibov_df.columns:
            ibov_df = ibov_df[ibov_df['benchmark'] == 'IBOV'][['dt', 'value']].sort_values('dt')
    else:
        ibov_df = pd.read_sql_query(
            "SELECT dt, value FROM benchmark_daily WHERE benchmark='IBOV' ORDER BY dt", conn
        )
    if not ibov_df.empty:
        ibov_df['dt'] = pd.to_datetime(ibov_df['dt'])

    if conn:
        conn.close()

    # Historico (metadados: nome, gestora, ANBIMA, PL)
    hist_df = pd.DataFrame()
    if os.path.exists(HIST_PATH):
        hist = pickle.load(open(HIST_PATH, 'rb'))
        hist_df = pd.DataFrame(hist)

    return features_df, results_df, sector_df, holdings_df, viz_df, cotas_df, ibov_df, hist_df, metadata_df


@st.cache_data(ttl=3600)
def load_cotas(cnpjs):
    """Carrega cotas diarias sob demanda para CNPJs especificos."""
    if not cnpjs:
        return pd.DataFrame()
    if _USE_PARQUET:
        path = os.path.join(DATA_DIR, 'cotas_diarias.parquet')
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_parquet(path, filters=[('cnpj_clean', 'in', list(cnpjs))])
        df = df.sort_values('dt_comptc')
    else:
        conn = sqlite3.connect(DB_PATH)
        placeholders = ','.join(['?'] * len(cnpjs))
        query = f'SELECT cnpj_clean, dt_comptc, vl_quota FROM cotas_diarias WHERE cnpj_clean IN ({placeholders}) ORDER BY dt_comptc'
        df = pd.read_sql_query(query, conn, params=list(cnpjs))
        conn.close()
    if not df.empty:
        df['dt_comptc'] = pd.to_datetime(df['dt_comptc'])
    return df


def get_fund_name(cnpj, hist_df):
    """Retorna o nome curto do fundo a partir do CNPJ."""
    if hist_df.empty:
        return cnpj
    match = hist_df[hist_df['cnpj_clean'] == cnpj]
    if not match.empty:
        nome = match.iloc[0].get('nome', cnpj)
        # Encurtar nome
        if len(nome) > 50:
            nome = nome[:50] + '...'
        return nome
    return cnpj


# ============================================================
# PAGINA 1: VISAO GERAL
# ============================================================
def page_visao_geral(features_df, results_df, hist_df):
    st.header("📊 Visao Geral dos Clusters")

    n_clusters = results_df['cluster_id'].nunique()
    n_funds = len(results_df)
    avg_per_cluster = n_funds / n_clusters if n_clusters > 0 else 0

    # Peer groups da hist_df
    n_peer_groups = 0
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        clustered_cnpjs = set(results_df['cnpj_clean'])
        hist_clustered = hist_df[hist_df['cnpj_clean'].isin(clustered_cnpjs)]
        n_peer_groups = hist_clustered['peer_group'].nunique()

    # Metricas com contexto
    avg_vol = features_df['ann_volatility'].mean() if 'ann_volatility' in features_df.columns else None
    avg_sharpe = features_df['sharpe_approx'].mean() if 'sharpe_approx' in features_df.columns else None
    avg_beta = features_df['beta_ibov'].median() if 'beta_ibov' in features_df.columns else None

    cols = st.columns(5)
    with cols[0]:
        metric_card("Total de Fundos", f"{n_funds:,}",
                    subtitle=f"Vol media: {avg_vol:.1%}" if avg_vol else None)
    with cols[1]:
        metric_card("Clusters", str(n_clusters),
                    subtitle=f"Media: {avg_per_cluster:.0f} fundos/cluster")
    with cols[2]:
        metric_card("Sharpe Medio", f"{avg_sharpe:.2f}" if avg_sharpe else "-",
                    subtitle=f"Beta mediano: {avg_beta:.2f}" if avg_beta else None)
    with cols[3]:
        metric_card("Classificacoes TAG", str(n_peer_groups))
    with cols[4]:
        silhouette = features_df.attrs.get('silhouette', '-')
        metric_card("Silhouette Score", f"{silhouette}")

    # ── Funil: Como chegamos nessa base (v7: 4 etapas) ──
    if not hist_df.empty:
        _total_base = len(hist_df)

        # Contagem pos-filtro cotistas: total - removidos (com dados E <20)
        # O filtro real remove apenas fundos COM dados de cotistas que tem < 20.
        # Fundos SEM dados de cotistas sao mantidos.
        _n_pos_cotistas = _total_base  # fallback: sem filtro
        try:
            if _USE_PARQUET:
                _cot_path = os.path.join(DATA_DIR, 'cotistas_diario.parquet')
                if os.path.exists(_cot_path):
                    _cot_all = pd.read_parquet(_cot_path)
                    _cotistas_df = _cot_all.loc[
                        _cot_all.groupby('cnpj_clean')['dt_comptc'].idxmax(),
                        ['cnpj_clean', 'nr_cotst']
                    ]
                else:
                    _cotistas_df = pd.DataFrame()
            else:
                _conn = sqlite3.connect(DB_PATH)
                _cotistas_df = pd.read_sql_query(
                    """SELECT cnpj_clean, nr_cotst
                       FROM cotistas_diario
                       WHERE (cnpj_clean, dt_comptc) IN (
                           SELECT cnpj_clean, MAX(dt_comptc)
                           FROM cotistas_diario GROUP BY cnpj_clean
                       )""", _conn
                )
                _conn.close()
            # Fundos com dados E < 20 cotistas = removidos
            _all_cnpjs_hist = set(hist_df['cnpj_clean'])
            _cotistas_map = dict(zip(_cotistas_df['cnpj_clean'], _cotistas_df['nr_cotst']))
            _n_removidos_gaveta = sum(
                1 for c in _all_cnpjs_hist
                if c in _cotistas_map and _cotistas_map[c] < 20
            )
            _n_pos_cotistas = _total_base - _n_removidos_gaveta
        except Exception:
            if 'nr_cotistas' in hist_df.columns:
                _n_removidos_gaveta = ((hist_df['nr_cotistas'].notna()) & (hist_df['nr_cotistas'] < 20)).sum()
                _n_pos_cotistas = _total_base - _n_removidos_gaveta

        # Representativos (pos-deduplicacao)
        _n_representativo = 0
        if 'is_representative' in hist_df.columns:
            _n_representativo = hist_df['is_representative'].sum()

        # Situacao dos representativos (para info no passo 3)
        _n_ativos_repr = 0
        _n_inativos_repr = 0
        if 'status_base' in hist_df.columns and 'is_representative' in hist_df.columns:
            _repr_df = hist_df[hist_df['is_representative'] == True]
            _n_ativos_repr = (_repr_df['status_base'] == 'ATIVO').sum()
            _n_inativos_repr = _n_representativo - _n_ativos_repr

        # Base final: apenas fundos ativos clusterizados
        _n_final = n_funds

        # Montar etapas do funil (v8: 4 etapas — ativos como ultimo filtro)
        _steps = []
        _steps.append(("Base CVM completa (FIA)", _total_base, TAG_OFFWHITE,
                       "Todos os fundos FIA registrados na CVM (ativos + inativos)"))
        if _n_pos_cotistas < _total_base:
            _steps.append(("Min. 20 cotistas", _n_pos_cotistas, "#C490F5",
                           "Remove fundos de gaveta / exclusivos com menos de 20 cotistas (padrao ANBIMA/IHFA)"))
        if _n_representativo > 0:
            _info_repr = f"{_n_ativos_repr:,} ativos + {_n_inativos_repr:,} inativos" if _n_ativos_repr > 0 else ""
            _steps.append(("Representativo (1 por estrategia)", _n_representativo, "#58C6F5",
                           f"Remove feeders, masters duplicados e veiculos secundarios. {_info_repr}"))
        _steps.append(("Fundos ativos", _n_final, TAG_LARANJA,
                       f"Apenas fundos em funcionamento normal. {n_clusters} clusters gerados."))

        # Construir HTML do funil
        _max_val = _steps[0][1] if _steps else 1
        _funnel_rows = ""
        for _i, (_label, _val, _color, _desc) in enumerate(_steps):
            _bar_pct = max((_val / _max_val) * 100, 8) if _max_val > 0 else 8
            _is_last = _i == len(_steps) - 1
            _border = f"border:2px solid {TAG_LARANJA};" if _is_last else ""
            _bg = f"{TAG_LARANJA}15" if _is_last else f"{TAG_BG_CARD}"
            _arrow = ""
            if _i > 0:
                _prev_val = _steps[_i - 1][1]
                _removed = _prev_val - _val
                if _removed > 0:
                    _arrow = (f'<div style="text-align:center;color:{TEXT_MUTED};font-size:11px;'
                              f'margin:2px 0;line-height:1;">▼ <span style="color:#ED5A6E;">-{_removed:,}</span></div>')

            _funnel_rows += f"""{_arrow}
            <div style="background:{_bg};{_border}border-radius:8px;padding:10px 16px;display:flex;align-items:center;gap:14px;">
                <div style="min-width:50px;text-align:right;">
                    <span style="font-size:20px;font-weight:800;color:{_color};">{_val:,}</span>
                </div>
                <div style="flex:1;">
                    <div style="background:{CHART_GRID};border-radius:4px;height:14px;overflow:hidden;margin-bottom:4px;">
                        <div style="width:{_bar_pct:.0f}%;height:100%;background:{_color};border-radius:4px;
                                    transition:width 0.3s;"></div>
                    </div>
                    <div style="font-size:12px;font-weight:600;color:{TAG_OFFWHITE};">{_label}</div>
                    <div style="font-size:10px;color:{TEXT_MUTED};margin-top:1px;">{_desc}</div>
                </div>
            </div>"""

        with st.expander("📋 Como chegamos nessa base?", expanded=True):
            st.markdown(
                f'<div style="display:flex;flex-direction:column;gap:0px;margin:8px 0;">'
                f'{_funnel_rows}'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Pipeline: Base CVM completa \u2192 filtra fundos de gaveta (min. 20 cotistas) \u2192 "
                "deduplica veiculos da mesma estrategia (feeders\u2192master) \u2192 "
                "seleciona apenas fundos ativos \u2192 "
                "clustering hierarquico 2-estagios com 16 features de retorno + holdings. "
                "Inativos preservados no historico para estudos de survivorship bias."
            )

    st.markdown("---")

    # ---- Distribuicao por Classificacao (peer_group) ----
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        st.subheader("Distribuicao por Classificacao")
        st.caption("**Classificacao** = categorias definidas pela TAG (Base Geral), como RV Valor, Long Biased, Small Caps, etc. Reflete a estrategia e mandato do fundo.")

        clustered_cnpjs = set(results_df['cnpj_clean'])
        hist_clustered = hist_df[hist_df['cnpj_clean'].isin(clustered_cnpjs)]
        pg_counts = hist_clustered.groupby('peer_group').size().reset_index(name='count')
        pg_counts = pg_counts.sort_values('count', ascending=False)

        if not pg_counts.empty:
            pg_colors = [PG_COLORS.get(pg, '#6A6864') for pg in pg_counts['peer_group']]

            fig = go.Figure(go.Treemap(
                labels=pg_counts['peer_group'],
                parents=[''] * len(pg_counts),
                values=pg_counts['count'],
                textinfo='label+value',
                marker=dict(
                    colors=pg_colors,
                    line=dict(color=TAG_BG_DARK, width=2),
                ),
                textfont=dict(size=14, color=TAG_OFFWHITE),
            ))
            fig.update_layout(**_chart_layout("", height=400, showlegend=False))
            st.plotly_chart(fig, width="stretch")

            # Tabela peer_group com stats
            pg_merged = hist_clustered[['cnpj_clean', 'peer_group']].merge(
                features_df, on='cnpj_clean', how='inner'
            )
            if not pg_merged.empty:
                pg_stats = pg_merged.groupby('peer_group').agg({
                    'cnpj_clean': 'count',
                    'beta_ibov': 'mean',
                    'ann_volatility': 'mean',
                    'corr_ibov': 'mean',
                    'sharpe_approx': 'mean',
                    'smll_affinity': 'mean',
                }).round(3)
                pg_stats.columns = ['Fundos', 'Beta IBOV', 'Volatilidade', 'Corr IBOV', 'Sharpe', 'Afinidade SMLL']
                pg_stats = pg_stats.sort_values('Fundos', ascending=False)
                st.dataframe(pg_stats, width="stretch",
                    column_config={
                        "Fundos": st.column_config.NumberColumn("Fundos", format="%d"),
                        "Beta IBOV": st.column_config.NumberColumn("Beta IBOV", format="%.3f"),
                        "Volatilidade": st.column_config.ProgressColumn("Volatilidade",
                            min_value=0, max_value=0.5, format="%.1%%"),
                        "Corr IBOV": st.column_config.ProgressColumn("Corr IBOV",
                            min_value=0, max_value=1.0, format="%.3f"),
                        "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.3f"),
                        "Afinidade SMLL": st.column_config.NumberColumn("Afin. SMLL", format="%.3f"),
                    })

        # Legenda detalhada das classificacoes
        _pg_descriptions = {
            'RV Valor/Retorno Absoluto': ('Fundos de stock picking com mandato amplo. Buscam retorno absoluto selecionando acoes individualmente, '
                                          'sem compromisso de seguir um indice. Beta e correlacao com IBOV variam bastante.'),
            'RV Long Biased': ('Fundos que podem reduzir exposicao liquida a acoes (operam vendido parcialmente). '
                               'Tendem a ter beta < 1 e menor correlacao com IBOV, oferecendo protecao parcial em quedas.'),
            'RV Small Caps': ('Fundos focados em empresas de menor capitalizacao (small/mid caps). '
                              'Alta afinidade com indice SMLL, maior volatilidade e menor liquidez. Potencial de retorno elevado.'),
            'RV Indexado Ativo': ('Fundos que buscam superar um indice de referencia (IBOV, IBrX) com gestao ativa. '
                                 'Beta e correlacao altos com IBOV, mas com alpha positivo quando bem geridos.'),
            'RV Indexado Passivo': ('Fundos que replicam fielmente um indice (IBOV, IBrX). Beta ≈ 1, correlacao ≈ 1 com IBOV. '
                                   'Tracking error muito baixo. Custos menores que gestao ativa.'),
            'RV Dividendos': ('Fundos focados em acoes pagadoras de dividendos. '
                              'Correlacao alta com indice IDIV, perfil mais defensivo, volatilidade geralmente menor.'),
            'Inv. Exterior': ('Fundos que investem majoritariamente em acoes internacionais. '
                              'Baixa correlacao com IBOV (diversificacao geografica). Expostos a variacao cambial.'),
            'Long Short': ('Fundos que operam comprados e vendidos simultaneamente, buscando retorno pela diferenca entre posicoes. '
                           'Beta proximo de zero, baixa correlacao com IBOV. Retorno depende da selecao de pares.'),
            'ESG': ('Fundos com mandato de investimento sustentavel (ambiental, social e governanca). '
                    'Filtram empresas por criterios ESG alem da analise financeira tradicional.'),
            'RV Pipe': ('Fundos especializados em operacoes de PIPE (Private Investment in Public Equity). '
                        'Investem em ofertas restritas de acoes ja listadas, geralmente com desconto.'),
        }

        # Montar HTML da legenda com cores
        _legend_items = ""
        for _pg_name in pg_counts['peer_group'].tolist():
            _pg_color = PG_COLORS.get(_pg_name, '#6A6864')
            _pg_desc = _pg_descriptions.get(_pg_name, '')
            if _pg_desc:
                _legend_items += (
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">'
                    f'<span style="display:inline-block;min-width:14px;width:14px;height:14px;border-radius:3px;'
                    f'background:{_pg_color};margin-top:2px;flex-shrink:0;"></span>'
                    f'<div><span style="font-weight:700;color:{TAG_OFFWHITE};font-size:13px;">{_pg_name}</span>'
                    f'<br><span style="color:{TEXT_MUTED};font-size:12px;">{_pg_desc}</span></div>'
                    f'</div>'
                )

        if _legend_items:
            st.markdown(
                f'<div style="background:{TAG_BG_CARD};border-radius:10px;padding:18px 20px;'
                f'border:1px solid {TAG_VERMELHO}30;margin-top:16px;">'
                f'<div style="font-size:11px;font-weight:700;color:{TAG_LARANJA};text-transform:uppercase;'
                f'letter-spacing:0.8px;margin-bottom:14px;">Legenda das Classificacoes</div>'
                f'{_legend_items}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # ---- Distribuicao por Cluster (dentro de expander) ----
    with st.expander("Distribuicao por Cluster (agrupamento quantitativo)", expanded=False):
        st.caption("**Cluster** = agrupamento automatico baseado em 18 features de retorno (beta, volatilidade, correlacao, etc.). "
                   "Fundos no mesmo cluster tem comportamento quantitativo semelhante. Um cluster pode conter fundos de classificacoes diferentes.")

        # Treemap
        cluster_counts = results_df.groupby(['cluster_id', 'cluster_label']).size().reset_index(name='count')
        cluster_counts = cluster_counts.sort_values('cluster_id')

        # Cores fixas por cluster_id
        _treemap_colors = [TAG_CHART_COLORS[(int(cid) - 1) % len(TAG_CHART_COLORS)] for cid in cluster_counts['cluster_id']]

        fig = go.Figure(go.Treemap(
            labels=cluster_counts['cluster_label'],
            parents=[''] * len(cluster_counts),
            values=cluster_counts['count'],
            textinfo='label+value',
            marker=dict(
                colors=_treemap_colors,
                line=dict(color=TAG_BG_DARK, width=2),
            ),
            textfont=dict(size=14, color=TAG_OFFWHITE),
        ))
        fig.update_layout(**_chart_layout("", height=400, showlegend=False))
        st.plotly_chart(fig, width="stretch")

        # Tabela de stats por cluster
        merged = results_df.merge(features_df, on='cnpj_clean', how='left')
        stats = merged.groupby('cluster_label').agg({
            'cnpj_clean': 'count',
            'beta_ibov': 'mean',
            'ann_volatility': 'mean',
            'corr_ibov': 'mean',
            'sharpe_approx': 'mean',
            'smll_affinity': 'mean',
            'idiv_affinity': 'mean',
        }).round(3)
        stats.columns = ['Fundos', 'Beta IBOV', 'Volatilidade', 'Corr IBOV', 'Sharpe', 'Afinidade SMLL', 'Afinidade IDIV']
        stats = stats.sort_values('Fundos', ascending=False)
        st.dataframe(stats, width="stretch",
            column_config={
                "Fundos": st.column_config.NumberColumn("Fundos", format="%d"),
                "Beta IBOV": st.column_config.NumberColumn("Beta IBOV", format="%.3f"),
                "Volatilidade": st.column_config.ProgressColumn("Volatilidade",
                    min_value=0, max_value=0.5, format="%.1%%"),
                "Corr IBOV": st.column_config.ProgressColumn("Corr IBOV",
                    min_value=0, max_value=1.0, format="%.3f"),
                "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.3f"),
                "Afinidade SMLL": st.column_config.NumberColumn("Afin. SMLL", format="%.3f"),
                "Afinidade IDIV": st.column_config.NumberColumn("Afin. IDIV", format="%.3f"),
            })


# Cores fixas por classificacao (peer_group)
PG_COLORS = {
    'RV Valor/Retorno Absoluto': '#FF8853',
    'Inv. Exterior': '#5C85F7',
    'RV Indexado Ativo': '#6BDE97',
    'RV Long Biased': '#FFBB00',
    'RV Small Caps': '#ED5A6E',
    'RV Dividendos': '#58C6F5',
    'RV Indexado Passivo': '#A485F2',
    'ESG': '#477C88',
    'Long Short': '#002A6E',
    'RV Pipe': '#FFD700',
}


# ============================================================
# PAGINA 2: MAPA DE CLUSTERS (PCA)
# ============================================================
def page_mapa_clusters(features_df, results_df, viz_df, hist_df, metadata_df=None):
    st.header("🗺️ Mapa de Fundos")

    # Toggle: colorir por Classificacao ou por Cluster
    color_by = st.radio(
        "Colorir por:",
        ["Classificacao TAG", "Cluster Quantitativo"],
        horizontal=True,
        key="mapa_color_by",
        help="**Classificacao TAG** = categorias definidas pela TAG (RV Valor, Long Biased, Small Caps, etc.). "
             "**Cluster Quantitativo** = agrupamento automatico baseado em features de retorno e carteira.",
    )

    # Usar coordenadas pre-computadas (UMAP ou PCA fallback)
    has_viz = not viz_df.empty and 'x' in viz_df.columns and 'y' in viz_df.columns

    if has_viz:
        # Merge viz_coords com cluster labels
        merged = viz_df.merge(
            results_df[['cnpj_clean', 'cluster_id', 'cluster_label']],
            on='cnpj_clean', how='inner',
            suffixes=('_viz', '')
        )
        if 'cluster_id_viz' in merged.columns:
            merged = merged.drop(columns=['cluster_id_viz'], errors='ignore')
        merged['X'] = merged['x']
        merged['Y'] = merged['y']
        viz_method = merged['method'].iloc[0] if 'method' in merged.columns else 'UMAP'
        viz_info = merged['info'].iloc[0] if 'info' in merged.columns else ''
    else:
        viz_method = 'PCA'
        merged = results_df[['cnpj_clean', 'cluster_id', 'cluster_label']].merge(
            features_df, on='cnpj_clean', how='inner'
        )
        feat_cols = [c for c in FEATURE_COLS if c in merged.columns]
        X = merged[feat_cols].copy()
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        merged['X'] = X_pca[:, 0]
        merged['Y'] = X_pca[:, 1]
        var_explained = pca.explained_variance_ratio_
        viz_info = f'PC1={var_explained[0]:.1%} PC2={var_explained[1]:.1%}'

    # Adicionar peer_group ao merged
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        pg_map = dict(zip(hist_df['cnpj_clean'], hist_df['peer_group']))
        merged['peer_group'] = merged['cnpj_clean'].map(pg_map).fillna('Outros')
    else:
        merged['peer_group'] = 'Outros'

    # Remover outliers extremos (|z-score| > 3.5)
    _x_z = (merged['X'] - merged['X'].mean()) / merged['X'].std()
    _y_z = (merged['Y'] - merged['Y'].mean()) / merged['Y'].std()
    _outlier_mask = (_x_z.abs() > 3.5) | (_y_z.abs() > 3.5)
    _n_outliers = _outlier_mask.sum()
    merged_plot = merged[~_outlier_mask].copy()

    # Nomes dos fundos
    merged_plot['nome'] = merged_plot['cnpj_clean'].apply(lambda c: get_fund_name(c, hist_df))

    # Merge features para hover rico
    _hover_feats = ['corr_ibov', 'beta_ibov', 'ann_volatility', 'sharpe_approx', 'ann_return']
    for _hf in _hover_feats:
        if _hf not in merged_plot.columns and _hf in features_df.columns:
            _feat_map = dict(zip(features_df['cnpj_clean'], features_df[_hf]))
            merged_plot[_hf] = merged_plot['cnpj_clean'].map(_feat_map)

    if _n_outliers > 0:
        st.caption(f"{_n_outliers} fundo(s) com valores extremos removido(s) da visualizacao")

    # ── FILTROS: selecionar categorias especificas ──
    if color_by == "Classificacao TAG":
        color_col = 'peer_group'
        all_categories = sorted(merged_plot[color_col].dropna().unique())
    else:
        color_col = 'cluster_label'
        all_categories = sorted(merged_plot[color_col].dropna().unique())

    col_filt1, col_filt2 = st.columns([1, 1])
    with col_filt1:
        selected_cats = st.multiselect(
            f"Filtrar {'classificacoes' if color_by == 'Classificacao TAG' else 'clusters'}:",
            options=all_categories,
            default=[],
            key="mapa_filter_cats",
            placeholder="Todos (clique para filtrar)",
            help="Selecione categorias especificas para exibir no mapa. Vazio = todas.",
        )

    # ── DESTAQUE: selecionar fundos para destacar ──
    with col_filt2:
        all_fund_names = sorted(merged_plot['nome'].dropna().unique())
        highlighted_funds = st.multiselect(
            "Destacar fundos:",
            options=all_fund_names,
            default=[],
            key="mapa_highlight_funds",
            placeholder="Buscar fundo por nome...",
            help="Selecione fundos para destacar no mapa. Eles aparecerao maiores e com borda.",
        )

    # Aplicar filtro de categorias
    if selected_cats:
        merged_plot = merged_plot[merged_plot[color_col].isin(selected_cats)]

    # Determinar categorias e ordem
    if color_by == "Classificacao TAG":
        cat_counts = merged_plot[color_col].value_counts()
        labels_unique = cat_counts.index.tolist()
    else:
        labels_unique = sorted(merged_plot[color_col].unique())

    # Mapear cluster_id -> cor fixa (para que a cor nao dependa da ordem/filtragem)
    _all_cluster_ids = sorted(merged['cluster_id'].dropna().unique())
    CLUSTER_COLOR_MAP = {cid: TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)]
                         for i, cid in enumerate(_all_cluster_ids)}
    # Mapear cluster_label -> cor via cluster_id
    _label_to_color = {}
    for cid in _all_cluster_ids:
        lbl = merged.loc[merged['cluster_id'] == cid, 'cluster_label'].iloc[0] if (merged['cluster_id'] == cid).any() else ''
        _label_to_color[lbl] = CLUSTER_COLOR_MAP[cid]

    # Separar fundos destacados vs regulares
    highlight_set = set(highlighted_funds) if highlighted_funds else set()
    has_highlights = len(highlight_set) > 0

    # Scatter plot
    fig = go.Figure()

    for i, lbl in enumerate(labels_unique):
        subset = merged_plot[merged_plot[color_col] == lbl]
        if color_by == "Classificacao TAG":
            color = PG_COLORS.get(lbl, TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)])
        else:
            color = _label_to_color.get(lbl, TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)])

        if has_highlights:
            # Fundos regulares (nao destacados) - menores e mais transparentes
            regular = subset[~subset['nome'].isin(highlight_set)]
            highlighted = subset[subset['nome'].isin(highlight_set)]

            if not regular.empty:
                _cdata = regular[['nome','cluster_label','peer_group'] +
                                 [f for f in _hover_feats if f in regular.columns]].values
                fig.add_trace(go.Scatter(
                    x=regular['X'], y=regular['Y'],
                    mode='markers',
                    name=f'{lbl} ({len(subset)})',
                    legendgroup=lbl,
                    marker=dict(color=color, size=7, opacity=0.45,
                                line=dict(width=0.5, color='rgba(255,255,255,0.3)')),
                    text=regular['nome'],
                    customdata=_cdata,
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Cluster: %{customdata[1]}<br>'
                        'Classif: %{customdata[2]}<br>'
                        'Corr: %{customdata[3]:.3f} | Beta: %{customdata[4]:.2f}<br>'
                        'Vol: %{customdata[5]:.1%} | Sharpe: %{customdata[6]:.3f}'
                        '<extra></extra>'
                    ),
                ))
            elif not highlighted.empty:
                # Se todos estao destacados, adicionar trace vazio para legenda
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    name=f'{lbl} ({len(subset)})',
                    legendgroup=lbl,
                    marker=dict(color=color, size=5),
                ))

            if not highlighted.empty:
                fig.add_trace(go.Scatter(
                    x=highlighted['X'], y=highlighted['Y'],
                    mode='markers+text',
                    name=f'★ {lbl}',
                    legendgroup=lbl,
                    showlegend=False,
                    marker=dict(
                        color=color, size=16, opacity=1.0,
                        line=dict(width=3, color=TAG_OFFWHITE),
                        symbol='star',
                    ),
                    text=highlighted['nome'],
                    textposition='top center',
                    textfont=dict(size=11, color=TAG_OFFWHITE, family='Inter'),
                    hovertemplate='<b>%{text}</b> ★ DESTAQUE<extra></extra>',
                ))
        else:
            _cdata = subset[['nome','cluster_label','peer_group'] +
                            [f for f in _hover_feats if f in subset.columns]].values
            fig.add_trace(go.Scatter(
                x=subset['X'], y=subset['Y'],
                mode='markers',
                name=f'{lbl} ({len(subset)})',
                marker=dict(color=color, size=9, opacity=0.85,
                            line=dict(width=0.8, color='rgba(255,255,255,0.4)')),
                text=subset['nome'],
                customdata=_cdata,
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'Cluster: %{customdata[1]}<br>'
                    'Classif: %{customdata[2]}<br>'
                    'Corr: %{customdata[3]:.3f} | Beta: %{customdata[4]:.2f}<br>'
                    'Vol: %{customdata[5]:.1%} | Sharpe: %{customdata[6]:.3f}'
                    '<extra></extra>'
                ),
            ))

    # Centroid labels with key metrics
    # Build a features lookup merged with cluster labels for metric computation
    _feat_with_labels = features_df.merge(
        results_df[['cnpj_clean', 'cluster_label']], on='cnpj_clean', how='inner'
    )
    # Also add peer_group for classification mode
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        _pg_map_centroid = dict(zip(hist_df['cnpj_clean'], hist_df['peer_group']))
        _feat_with_labels['peer_group'] = _feat_with_labels['cnpj_clean'].map(_pg_map_centroid).fillna('Outros')
    else:
        _feat_with_labels['peer_group'] = 'Outros'

    _centroids = merged_plot.groupby(color_col).agg({'X': 'median', 'Y': 'median'}).reset_index()
    for _, _crow in _centroids.iterrows():
        full_label = str(_crow[color_col])
        _clbl = full_label if len(full_label) <= 22 else full_label[:20] + '..'
        # Get metrics for this cluster/classification from the features data
        _grp_data = _feat_with_labels[_feat_with_labels[color_col] == full_label]
        if not _grp_data.empty and 'corr_ibov' in _grp_data.columns and 'ann_volatility' in _grp_data.columns:
            _med_corr = _grp_data['corr_ibov'].median()
            _med_vol = _grp_data['ann_volatility'].median()
            _ann_text = (f"<b>{_clbl}</b><br>"
                         f"<span style='font-size:8px; color:{TEXT_MUTED}'>corr={_med_corr:.2f} | vol={_med_vol:.0%}</span>")
        else:
            _ann_text = f"<b>{_clbl}</b>"
        fig.add_annotation(
            x=_crow['X'], y=_crow['Y'],
            text=_ann_text,
            showarrow=False,
            font=dict(color=TAG_OFFWHITE, size=10, family='Inter'),
            bgcolor='rgba(30,12,20,0.75)',
            bordercolor='rgba(230,228,219,0.2)',
            borderwidth=1,
            borderpad=3,
        )

    if viz_method == 'UMAP':
        title = f"UMAP 2D ({viz_info})"
        x_title = 'Correlacao com mercado  →'
        y_title = '←  Agressividade / Small Cap'
    else:
        title = f"PCA 2D ({viz_info})"
        x_title = 'PC1'
        y_title = 'PC2'

    layout = _chart_layout(title, height=CHART_HEIGHT_FULL)
    layout['xaxis']['title'] = dict(text=x_title, font=dict(size=11, color=TEXT_MUTED))
    layout['yaxis']['title'] = dict(text=y_title, font=dict(size=11, color=TEXT_MUTED))
    if viz_method == 'UMAP':
        layout['xaxis']['showticklabels'] = False
        layout['yaxis']['showticklabels'] = False
    # Legenda horizontal embaixo
    layout['legend'] = dict(
        orientation='h',
        yanchor='top', y=-0.06,
        xanchor='center', x=0.5,
        bgcolor='rgba(0,0,0,0)',
        font=dict(color=TAG_OFFWHITE, size=10),
        itemwidth=30,
    )
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")

    # ── Info do filtro ──
    _filter_info = f"Exibindo **{len(merged_plot):,}** fundos"
    if selected_cats:
        _filter_info += f" ({len(selected_cats)} {'classificacoes' if color_by == 'Classificacao TAG' else 'clusters'} selecionados)"
    if has_highlights:
        _filter_info += f" | **{len(highlight_set)} fundo(s) destacado(s)** com estrela ★"
    st.caption(_filter_info)

    # ── Nota interpretativa dos eixos ──
    if viz_method == 'UMAP':
        _axes_note = (
            f'<div style="background:{TAG_BG_CARD};border-radius:8px;padding:12px 16px;margin:8px 0 16px 0;'
            f'border-left:3px solid {TAG_LARANJA}40;">'
            f'<div style="color:{TAG_OFFWHITE};font-size:0.82rem;font-weight:600;margin-bottom:6px;">Como ler este mapa</div>'
            f'<div style="color:{TEXT_MUTED};font-size:0.78rem;line-height:1.5;">'
            f'<b style="color:{TAG_OFFWHITE};">Eixo horizontal</b>: correlacao com o mercado. '
            f'Quanto mais a direita, maior a correlacao com o Ibovespa (fundos direcionais e indexados). '
            f'A esquerda ficam os fundos descorrelacionados (protecao e long/short).<br>'
            f'<b style="color:{TAG_OFFWHITE};">Eixo vertical</b>: agressividade. '
            f'Quanto mais abaixo, maior o beta, volatilidade e exposicao a small caps. '
            f'No topo ficam os fundos mais conservadores e indexados.<br>'
            f'<span style="color:{TEXT_MUTED};font-style:italic;">Fundos proximos no mapa tem comportamento similar '
            f'(baseado em 15 metricas: retorno, risco, correlacoes e composicao de carteira).</span>'
            f'</div></div>'
        )
        st.markdown(_axes_note, unsafe_allow_html=True)

    # ── SCATTER 3D: Features reais (corr x vol x beta) ──
    st.markdown("---")
    st.subheader("Mapa 3D: Correlacao x Volatilidade x Beta")

    _3d_feats = {'corr_ibov': 'Correlacao IBOV', 'ann_volatility': 'Volatilidade', 'beta_ibov': 'Beta IBOV'}
    _has_3d = all(f in features_df.columns for f in _3d_feats.keys())

    if _has_3d:
        _3d_data = merged_plot.copy()
        _extra_3d = ['corr_ibov', 'ann_volatility', 'beta_ibov', 'sharpe_approx']
        for _f3 in _extra_3d:
            if _f3 not in _3d_data.columns and _f3 in features_df.columns:
                _fmap = dict(zip(features_df['cnpj_clean'], features_df[_f3]))
                _3d_data[_f3] = _3d_data['cnpj_clean'].map(_fmap)
        _3d_data = _3d_data.dropna(subset=['corr_ibov', 'ann_volatility', 'beta_ibov'])

        # Tamanho do marcador proporcional ao Sharpe (maior Sharpe = bolha maior)
        _sharpe_vals = _3d_data['sharpe_approx'].fillna(0).clip(0, 3)
        _3d_data['_marker_size'] = 3 + (_sharpe_vals / 3) * 7  # range 3-10

        fig3d = go.Figure()
        for lbl in labels_unique:
            _sub3d = _3d_data[_3d_data[color_col] == lbl]
            if _sub3d.empty:
                continue
            if color_by == "Classificacao TAG":
                _c3d = PG_COLORS.get(lbl, TAG_CHART_COLORS[labels_unique.index(lbl) % len(TAG_CHART_COLORS)])
            else:
                _c3d = _label_to_color.get(lbl, TAG_CHART_COLORS[labels_unique.index(lbl) % len(TAG_CHART_COLORS)])

            _sharpe_text = [f'{s:.2f}' if pd.notna(s) else '-' for s in _sub3d['sharpe_approx']]

            fig3d.add_trace(go.Scatter3d(
                x=_sub3d['corr_ibov'],
                y=_sub3d['ann_volatility'],
                z=_sub3d['beta_ibov'],
                mode='markers',
                name=f'{lbl} ({len(_sub3d)})',
                marker=dict(
                    color=_c3d,
                    size=_sub3d['_marker_size'],
                    opacity=0.8,
                    line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
                ),
                text=_sub3d['nome'],
                customdata=list(zip(_sharpe_text, _sub3d['cluster_label'])),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    'Cluster: %{customdata[1]}<br>'
                    'Correlacao: %{x:.2f}<br>'
                    'Volatilidade: %{y:.1%}<br>'
                    'Beta: %{z:.2f}<br>'
                    'Sharpe: %{customdata[0]}'
                    '<extra></extra>'
                ),
            ))

        # Camera com angulo levemente inclinado para melhor visao inicial
        _camera = dict(
            eye=dict(x=1.6, y=1.6, z=0.9),
            center=dict(x=0, y=0, z=-0.1),
        )

        _axis_style = dict(
            gridcolor='rgba(255,255,255,0.08)',
            backgroundcolor='rgba(0,0,0,0)',
            showbackground=True,
            zerolinecolor='rgba(255,136,83,0.3)',
        )

        fig3d.update_layout(
            scene=dict(
                xaxis=dict(
                    title=dict(text='Correlacao IBOV', font=dict(size=12, color=TAG_OFFWHITE)),
                    tickfont=dict(size=9, color=TEXT_MUTED),
                    range=[-0.1, 1.05],
                    **_axis_style,
                ),
                yaxis=dict(
                    title=dict(text='Volatilidade', font=dict(size=12, color=TAG_OFFWHITE)),
                    tickfont=dict(size=9, color=TEXT_MUTED),
                    tickformat='.0%',
                    **_axis_style,
                ),
                zaxis=dict(
                    title=dict(text='Beta', font=dict(size=12, color=TAG_OFFWHITE)),
                    tickfont=dict(size=9, color=TEXT_MUTED),
                    **_axis_style,
                ),
                bgcolor='rgba(0,0,0,0)',
                camera=_camera,
                aspectmode='cube',
            ),
            height=700,
            paper_bgcolor=TAG_BG_DARK,
            plot_bgcolor=TAG_BG_DARK,
            font=dict(color=TAG_OFFWHITE, family='Inter'),
            legend=dict(
                orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5,
                bgcolor='rgba(0,0,0,0)', font=dict(color=TAG_OFFWHITE, size=11),
                itemsizing='constant',
            ),
            margin=dict(l=0, r=0, t=10, b=40),
        )
        st.plotly_chart(fig3d, width="stretch")

        _3d_note = (
            f'<div style="background:{TAG_BG_CARD};border-radius:8px;padding:12px 16px;margin:4px 0 16px 0;'
            f'border-left:3px solid {TAG_LARANJA}40;">'
            f'<div style="color:{TAG_OFFWHITE};font-size:0.82rem;font-weight:600;margin-bottom:6px;">Como ler este grafico</div>'
            f'<div style="color:{TEXT_MUTED};font-size:0.78rem;line-height:1.6;">'
            f'Cada eixo mostra uma metrica real do fundo — sem projecoes artificiais.<br>'
            f'<b style="color:{TAG_OFFWHITE};">Correlacao</b> (X): quanto o fundo acompanha o Ibovespa (0 = nenhuma, 1 = perfeita).<br>'
            f'<b style="color:{TAG_OFFWHITE};">Volatilidade</b> (Y): risco anualizado — quanto maior, mais oscila.<br>'
            f'<b style="color:{TAG_OFFWHITE};">Beta</b> (Z): sensibilidade ao mercado (>1 = amplifica movimentos, <1 = amortece).<br>'
            f'<b style="color:{TAG_OFFWHITE};">Tamanho da bolha</b>: Sharpe ratio — bolhas maiores = melhor retorno ajustado ao risco.<br>'
            f'<span style="font-style:italic;">Arraste para girar. Scroll para zoom. Duplo-clique para resetar.</span>'
            f'</div></div>'
        )
        st.markdown(_3d_note, unsafe_allow_html=True)
    else:
        st.info("Features insuficientes para o mapa 3D.")

    # ── LEGENDA DETALHADA DAS CATEGORIAS ──
    _pg_descriptions = {
        'RV Valor/Retorno Absoluto': 'Fundos de stock picking com mandato amplo. Buscam retorno absoluto selecionando acoes individualmente, sem compromisso de seguir indice.',
        'RV Long Biased': 'Podem reduzir exposicao liquida a acoes (operam vendido parcialmente). Beta < 1 e menor correlacao com IBOV, protecao parcial em quedas.',
        'RV Small Caps': 'Focados em empresas de menor capitalizacao (small/mid caps). Alta afinidade com SMLL, maior volatilidade e potencial de retorno elevado.',
        'RV Indexado Ativo': 'Buscam superar um indice (IBOV, IBrX) com gestao ativa. Beta e correlacao altos, gerando alpha quando bem geridos.',
        'RV Indexado Passivo': 'Replicam fielmente um indice (IBOV, IBrX). Beta ≈ 1, correlacao ≈ 1, tracking error baixo. Custos menores.',
        'RV Dividendos': 'Focados em acoes pagadoras de dividendos. Correlacao alta com IDIV, perfil mais defensivo e volatilidade menor.',
        'Inv. Exterior': 'Investem majoritariamente em acoes internacionais. Baixa correlacao com IBOV (diversificacao geografica). Expostos a cambio.',
        'Long Short': 'Operam comprados e vendidos simultaneamente. Beta proximo de zero, baixa correlacao com IBOV. Retorno pela diferenca entre posicoes.',
        'ESG': 'Mandato de investimento sustentavel (ambiental, social e governanca). Filtram empresas por criterios ESG.',
        'RV Pipe': 'Especializados em PIPE (Private Investment in Public Equity). Investem em ofertas restritas de acoes listadas.',
    }

    # Montar legenda HTML
    st.markdown("---")
    if color_by == "Classificacao TAG":
        _legend_title = "Legenda: Classificacoes TAG"
        _legend_subtitle = ("As **Classificacoes TAG** sao categorias definidas manualmente com base na estrategia declarada do fundo, "
                           "alinhadas com a Base Geral (ANBIMA/CVM). Cada fundo pertence a exatamente uma classificacao.")
        _labels_to_show = labels_unique
        _get_color = lambda lbl, i: PG_COLORS.get(lbl, TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)])
    else:
        _legend_title = "Legenda: Clusters Quantitativos"
        _legend_subtitle = ("Os **Clusters Quantitativos** sao agrupamentos automaticos baseados em features de retorno e composicao de carteira "
                           "(beta, volatilidade, correlacao, afinidade small cap/dividendos, concentracao HHI, etc.). "
                           "O label de cada cluster descreve suas caracteristicas dominantes.")
        _labels_to_show = labels_unique
        _get_color = lambda lbl, i: _label_to_color.get(lbl, TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)])

    _legend_items = ""
    for _i, _lbl in enumerate(_labels_to_show):
        _color = _get_color(_lbl, _i)
        # Para classificacoes, usar _pg_descriptions; para clusters, usar metadata_df
        if color_by == "Classificacao TAG":
            _desc = _pg_descriptions.get(_lbl, '')
        else:
            _desc = _get_cluster_description(_lbl, metadata_df)
        if not _desc:
            _desc = 'Grupo de fundos com perfil quantitativo semelhante.'
        _count = len(merged_plot[merged_plot[color_col] == _lbl])
        # Tactical note (only for clusters)
        _tac_html = ''
        if color_by != "Classificacao TAG":
            _tac = _get_tactical_note(_lbl, metadata_df)
            if _tac:
                _tac_html = f'<br><span style="color:{TAG_LARANJA};font-size:11px;font-style:italic;">💡 {_tac}</span>'
        _legend_items += (
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px;">'
            f'<span style="display:inline-block;min-width:14px;width:14px;height:14px;border-radius:3px;'
            f'background:{_color};margin-top:3px;flex-shrink:0;"></span>'
            f'<div style="flex:1;"><span style="font-weight:700;color:{TAG_OFFWHITE};font-size:13px;">'
            f'{_lbl}</span> <span style="color:{TEXT_MUTED};font-size:11px;">({_count} fundos)</span>'
            f'<br><span style="color:{TEXT_MUTED};font-size:12px;">{_desc}</span>{_tac_html}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:{TAG_BG_CARD};border-radius:10px;padding:18px 20px;'
        f'border:1px solid {TAG_VERMELHO}30;">'
        f'<div style="font-size:11px;font-weight:700;color:{TAG_LARANJA};text-transform:uppercase;'
        f'letter-spacing:0.8px;margin-bottom:6px;">{_legend_title}</div>'
        f'<div style="font-size:12px;color:{TEXT_MUTED};margin-bottom:14px;">{_legend_subtitle}</div>'
        f'{_legend_items}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Explicacao da diferenca entre classificacao e cluster
    st.markdown(
        f'<div style="background:{TAG_BG_DARK};border-radius:8px;padding:14px 18px;'
        f'border:1px solid {CHART_GRID};margin-top:12px;">'
        f'<div style="font-size:11px;font-weight:700;color:{TAG_LARANJA};margin-bottom:8px;">💡 CLASSIFICACAO vs CLUSTER: Qual a diferenca?</div>'
        f'<div style="font-size:12px;color:{TEXT_MUTED};line-height:1.6;">'
        f'<b style="color:{TAG_OFFWHITE};">Classificacao TAG</b> = categoria definida pela TAG com base na estrategia '
        f'declarada do fundo (mandato, benchmark, estilo). E fixa e qualitativa.<br>'
        f'<b style="color:{TAG_OFFWHITE};">Cluster Quantitativo</b> = agrupamento automatico por similaridade '
        f'de comportamento (retornos, beta, volatilidade, composicao da carteira). Fundos da MESMA classificacao '
        f'podem cair em clusters DIFERENTES se seus perfis quantitativos divergirem — e vice-versa.'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PAGINA 3: PERFIL DO CLUSTER
# ============================================================
def page_perfil_cluster(features_df, results_df, sector_df, hist_df, metadata_df=None):
    st.header("🎯 Perfil do Cluster")

    merged = results_df[['cnpj_clean', 'cluster_id', 'cluster_label']].merge(
        features_df, on='cnpj_clean', how='inner'
    )

    # Adicionar peer_group
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        pg_map = dict(zip(hist_df['cnpj_clean'], hist_df['peer_group']))
        merged['peer_group'] = merged['cnpj_clean'].map(pg_map).fillna('Outros')
    else:
        merged['peer_group'] = merged['cluster_label']

    # Toggle: agrupar por Classificacao ou Cluster
    group_by = st.radio(
        "Agrupar por:",
        ["Classificacao TAG", "Cluster Quantitativo"],
        horizontal=True,
        key="perfil_group_by",
        help="**Classificacao** = categorias TAG (RV Valor, Long Biased, Small Caps, etc.). "
             "**Cluster** = agrupamento quantitativo automatico baseado em features de retorno.",
    )

    if group_by == "Classificacao TAG":
        group_col = 'peer_group'
        # Ordenar por contagem (maiores primeiro)
        cat_counts = merged[group_col].value_counts()
        labels_unique = cat_counts.index.tolist()
        select_label = "Selecione uma Classificacao"
    else:
        group_col = 'cluster_label'
        labels_unique = sorted(merged[group_col].unique())
        select_label = "Selecione um Cluster"

    selected = st.selectbox(select_label, labels_unique)

    cluster_data = merged[merged[group_col] == selected]
    n_funds = len(cluster_data)

    st.markdown(f"**{n_funds} fundos** em **{selected}**")

    # ---- DNA Card (Card de Jogador) ----
    _meta_row = None
    if metadata_df is not None and not metadata_df.empty:
        _meta_match = metadata_df[metadata_df['cluster_label'] == selected]
        if not _meta_match.empty:
            _meta_row = _meta_match.iloc[0]

    if _meta_row is not None:
        def _gauge_pct(value, lo, hi):
            """Convert value to 0-100 pct clamped within [lo, hi]."""
            if pd.isna(value):
                return 0
            return max(0, min(100, (value - lo) / (hi - lo) * 100))

        _gauges = [
            ("Direcionalidade", _meta_row.get('median_corr_ibov', 0), 0, 1),
            ("Risco", _meta_row.get('median_ann_volatility', 0), 0, 0.50),
            ("Beta", _meta_row.get('median_beta_ibov', 0), 0, 1.5),
            ("Sharpe", _meta_row.get('median_sharpe', 0), 0, 2.5),
            ("Small Cap Tilt", _meta_row.get('median_smll_affinity', 0), -0.15, 0.10),
        ]

        _gauge_rows_html = ""
        for _g_label, _g_val, _g_lo, _g_hi in _gauges:
            _pct = _gauge_pct(_g_val, _g_lo, _g_hi)
            _display_val = f"{_g_val:.2f}" if pd.notna(_g_val) else "–"
            _gauge_rows_html += (
                f'<div style="display:flex;align-items:center;margin-bottom:8px;">'
                f'<span style="width:130px;font-size:0.82rem;color:{TEXT_MUTED};">{_g_label}</span>'
                f'<div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:8px;margin:0 10px;">'
                f'<div style="background:{TAG_LARANJA};height:100%;width:{_pct:.1f}%;border-radius:4px;"></div>'
                f'</div>'
                f'<span style="width:50px;text-align:right;font-size:0.82rem;color:{TAG_OFFWHITE};">{_display_val}</span>'
                f'</div>'
            )

        _up_cap = _meta_row.get('median_up_capture', None)
        _dn_cap = _meta_row.get('median_down_capture', None)
        _up_str = f"{_up_cap:.0%}" if pd.notna(_up_cap) else "–"
        _dn_str = f"{_dn_cap:.0%}" if pd.notna(_dn_cap) else "–"

        _tac_note = _meta_row.get('tactical_note', '')
        _auto_desc = _meta_row.get('auto_description', '')
        _tac_html = f'<div style="font-style:italic;color:{TAG_OFFWHITE};font-size:0.85rem;margin-top:12px;">{_tac_note}</div>' if _tac_note else ''
        _desc_html = f'<div style="color:{TEXT_MUTED};font-size:0.78rem;margin-top:6px;">{_auto_desc}</div>' if _auto_desc else ''

        _card_html = (
            f'<div style="background:{TAG_BG_CARD};border-radius:12px;padding:24px 28px;border-left:4px solid {TAG_LARANJA};margin-bottom:20px;">'
            f'<div style="color:{TAG_OFFWHITE};font-size:1.15rem;font-weight:600;margin-bottom:16px;">{selected}</div>'
            f'{_gauge_rows_html}'
            f'<div style="display:flex;gap:24px;margin-top:14px;">'
            f'<div><span style="color:{TEXT_MUTED};font-size:0.78rem;">Up Capture</span><br>'
            f'<span style="color:#22C55E;font-size:1.1rem;font-weight:600;">{_up_str}</span></div>'
            f'<div><span style="color:{TEXT_MUTED};font-size:0.78rem;">Down Capture</span><br>'
            f'<span style="color:#EF4444;font-size:1.1rem;font-weight:600;">{_dn_str}</span></div>'
            f'</div>'
            f'{_tac_html}'
            f'{_desc_html}'
            f'</div>'
        )
        st.markdown(_card_html, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # Radar chart das features
    with col1:
        st.subheader("Radar de Features")
        radar_feats = [f for f in RADAR_FEATURES if f in cluster_data.columns]
        labels_radar = [FEATURE_LABELS.get(f, f) for f in radar_feats]

        cluster_means_dict = {f: cluster_data[f].mean() for f in radar_feats}
        cluster_vals = _compute_radar_values(cluster_means_dict, radar_feats, merged)

        fig = _build_radar_chart([
            {'values': cluster_vals, 'labels': labels_radar, 'name': selected,
             'color': TAG_LARANJA, 'fill': True},
            {'values': [0.5] * len(radar_feats), 'labels': labels_radar, 'name': 'Media Geral',
             'color': TEXT_MUTED, 'fill': False, 'dash': 'dot'},
        ])
        st.plotly_chart(fig, width="stretch")
        st.caption("Radar comparando o perfil do cluster (laranja) com a media geral (linha pontilhada). Acima de 0.5 = supera a media; abaixo = fica aquem.")

    # Sector breakdown
    with col2:
        st.subheader("Composicao Setorial")
        if not sector_df.empty:
            cluster_cnpjs = set(cluster_data['cnpj_clean'].tolist())
            cluster_sectors = sector_df[sector_df['cnpj_clean'].isin(cluster_cnpjs)]

            if not cluster_sectors.empty:
                sector_agg = cluster_sectors.groupby('setor')['peso'].mean().sort_values(ascending=False)
                sector_agg = sector_agg[sector_agg > 0.01]

                fig = go.Figure(go.Pie(
                    labels=sector_agg.index.tolist(),
                    values=sector_agg.values,
                    hole=0.4,
                    marker=dict(colors=TAG_CHART_COLORS[:len(sector_agg)]),
                    textfont=dict(color=TAG_OFFWHITE, size=11),
                    textinfo='label+percent',
                ))
                fig.update_layout(**_chart_layout("", height=400, showlegend=False))
                n_with_data = len(cluster_sectors['cnpj_clean'].unique())
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Baseado em {n_with_data} fundos com dados de carteira")
            else:
                st.info("Nenhum fundo deste cluster possui dados de carteira (BLC_4)")
        else:
            st.info("Dados de carteira nao disponiveis")

    # ---- Feature Importance: o que torna esse grupo unico ----
    _fi_features = [f for f in FEATURE_COLS if f in cluster_data.columns and f in merged.columns]
    if _fi_features:
        global_means_fi = merged[_fi_features].mean()
        global_stds_fi = merged[_fi_features].std().replace(0, 1)
        cluster_means_fi = cluster_data[_fi_features].mean()
        z_scores_fi = (cluster_means_fi - global_means_fi) / global_stds_fi
        z_scores_fi = z_scores_fi.dropna()
        top_fi = z_scores_fi.abs().sort_values(ascending=False).head(7).index
        z_top = z_scores_fi[top_fi].sort_values()

        if len(z_top) > 0:
            st.subheader("O que torna esse grupo unico")
            fi_labels = [FEATURE_LABELS.get(f, f) for f in z_top.index]
            fi_colors = ['#22C55E' if v >= 0 else '#EF4444' for v in z_top.values]
            fig_fi = go.Figure(go.Bar(
                x=z_top.values,
                y=fi_labels,
                orientation='h',
                marker_color=fi_colors,
                text=[f"{v:+.2f}" for v in z_top.values],
                textposition='outside',
                textfont=dict(color=TAG_OFFWHITE, size=11),
            ))
            fig_fi.update_layout(
                **_chart_layout("", height=CHART_HEIGHT_SMALL, showlegend=False),
            )
            fig_fi.update_layout(
                xaxis_title="Z-score vs media geral",
                yaxis=dict(gridcolor=CHART_GRID, tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_fi, width="stretch")
            st.caption("Features com maior desvio em relacao a media geral de todos os fundos. "
                       "Verde = acima da media, vermelho = abaixo. Quanto maior o valor absoluto, mais o cluster se diferencia nessa dimensao.")

    # ---- Resumo de Carteira (holdings features) ----
    holdings_feats = ['pct_smallcap', 'pct_largecap', 'pct_commodities', 'pct_utilities', 'hhi_portfolio', 'turnover']
    has_holdings = any(f in cluster_data.columns and cluster_data[f].notna().any() for f in holdings_feats)
    if has_holdings:
        st.subheader("Perfil de Carteira (medias do cluster)")
        hcols = st.columns(6)
        for i, feat in enumerate(holdings_feats):
            if feat in cluster_data.columns:
                avg_val = cluster_data[feat].mean()
                if pd.notna(avg_val):
                    if feat.startswith('pct_') or feat == 'turnover':
                        fmt = f"{avg_val:.0%}"
                    elif feat == 'hhi_portfolio':
                        fmt = f"{avg_val:.3f}"
                    else:
                        fmt = f"{avg_val:.2f}"
                    with hcols[i % 6]:
                        metric_card(FEATURE_LABELS.get(feat, feat), fmt)
        n_with_holdings = cluster_data[holdings_feats[0]].notna().sum() if holdings_feats[0] in cluster_data.columns else 0
        st.caption(f"Medias baseadas em {n_with_holdings} fundos com dados de carteira CDA/BLC_4. Concentracao HHI: 0=diversificado, 1=concentrado.")

    # ---- Composicao cruzada (Classif dentro do Cluster, ou Clusters dentro da Classif) ----
    if group_by == "Cluster Quantitativo":
        # Mostrar quais classificacoes compõem este cluster
        cross_col = 'peer_group'
        cross_title = "Classificacoes neste Cluster"
        cross_caption = "Quais classificacoes TAG compõem este cluster quantitativo. Se houver varias, significa que fundos de estrategias diferentes tem perfil de retorno parecido."
    else:
        # Mostrar quais clusters compõem esta classificacao
        cross_col = 'cluster_label'
        cross_title = "Clusters nesta Classificacao"
        cross_caption = "Em quais clusters quantitativos os fundos desta classificacao foram agrupados. Se houver varios, significa que dentro da mesma classificacao existem perfis de retorno distintos."

    cross_comp = cluster_data.groupby(cross_col).size().reset_index(name='count')
    cross_comp = cross_comp.sort_values('count', ascending=False)
    if len(cross_comp) > 1:
        st.subheader(cross_title)
        cross_colors = [PG_COLORS.get(lbl, TAG_LARANJA) for lbl in cross_comp[cross_col]] if group_by == "Cluster Quantitativo" else [TAG_LARANJA] * len(cross_comp)
        fig = go.Figure(go.Bar(
            x=cross_comp[cross_col],
            y=cross_comp['count'],
            marker_color=cross_colors,
            text=cross_comp['count'],
            textposition='outside',
            textfont=dict(color=TAG_OFFWHITE),
        ))
        fig.update_layout(**_chart_layout("", height=CHART_HEIGHT_SMALL, showlegend=False))
        fig.update_layout(xaxis=dict(tickangle=-30))
        st.plotly_chart(fig, width="stretch")
        st.caption(cross_caption)

    # Lista de fundos
    st.subheader("Fundos neste Cluster")
    display_df = cluster_data[['cnpj_clean']].copy()
    display_df['Nome'] = display_df['cnpj_clean'].apply(lambda c: get_fund_name(c, hist_df))

    if not hist_df.empty:
        info_cols = ['cnpj_clean', 'gestora', 'classif_anbima', 'peer_group', 'pl']
        available = [c for c in info_cols if c in hist_df.columns]
        if available:
            info = hist_df[available].drop_duplicates(subset='cnpj_clean')
            display_df = display_df.merge(info, on='cnpj_clean', how='left')
            # Renomear peer_group para Classificacao
            if 'peer_group' in display_df.columns:
                display_df = display_df.rename(columns={'peer_group': 'Classificacao'})

    # Add features (retorno + holdings)
    display_feats = ['beta_ibov', 'ann_volatility', 'corr_ibov', 'sharpe_approx',
                     'smll_affinity', 'pct_smallcap', 'pct_commodities', 'pct_utilities', 'hhi_portfolio']
    for feat in display_feats:
        if feat in cluster_data.columns:
            vals = cluster_data[feat].values
            if feat.startswith('pct_'):
                display_df[FEATURE_LABELS.get(feat, feat)] = [f"{v:.0%}" if pd.notna(v) else '-' for v in vals]
            elif feat == 'hhi_portfolio':
                display_df[FEATURE_LABELS.get(feat, feat)] = [f"{v:.3f}" if pd.notna(v) else '-' for v in vals]
            else:
                display_df[FEATURE_LABELS.get(feat, feat)] = vals

    st.dataframe(display_df.drop(columns=['cnpj_clean']).head(50), width="stretch", height=400)
    st.caption("Lista dos fundos que compõem este cluster, com suas principais metricas e classificacao. Use para identificar fundos especificos dentro do grupo e comparar caracteristicas individuais.")


# ============================================================
# PAGINA 4: ANALISE DE FUNDO
# ============================================================
def page_analise_fundo(features_df, results_df, sector_df, cotas_df, ibov_df, hist_df, metadata_df=None):
    st.header("🔍 Analise de Fundo")

    # Busca por nome ou CNPJ
    merged = results_df.merge(features_df, on='cnpj_clean', how='inner')
    merged['nome'] = merged['cnpj_clean'].apply(lambda c: get_fund_name(c, hist_df))

    # Selectbox pesquisavel com todos os fundos
    all_options = [f"{row['nome']} ({row['cnpj_clean']})" for _, row in merged.iterrows()]
    all_options_sorted = sorted(all_options)

    selected = st.selectbox(
        "🔎 Buscar fundo (digite para filtrar)",
        options=[""] + all_options_sorted,
        index=0,
        key="analise_fundo_select",
        placeholder="Digite o nome ou CNPJ para pesquisar...",
    )

    if not selected:
        st.info("Selecione um fundo na lista acima. Digite para filtrar por nome ou CNPJ.")
        return

    # Encontrar o fundo selecionado
    _sel_cnpj = selected.split('(')[-1].rstrip(')')
    matches = merged[merged['cnpj_clean'] == _sel_cnpj]

    if matches.empty:
        st.warning("Fundo nao encontrado nos dados")
        return

    fund = matches.iloc[0]

    cnpj = fund['cnpj_clean']
    st.markdown(f"### {fund['nome']}")

    # Buscar peer_group e gestora da hist_df
    fund_pg = '-'
    fund_gestora = '-'
    if not hist_df.empty:
        hm = hist_df[hist_df['cnpj_clean'] == cnpj]
        if not hm.empty:
            fund_pg = hm.iloc[0].get('peer_group', '-')
            fund_gestora = hm.iloc[0].get('gestora', '-')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Classificacao TAG", str(fund_pg))
    with col2:
        metric_card("Cluster", fund.get('cluster_label', '-'))
    with col3:
        metric_card("Beta IBOV", f"{fund.get('beta_ibov', 0):.2f}")
    with col4:
        metric_card("Volatilidade", f"{fund.get('ann_volatility', 0):.1%}")

    if fund_gestora != '-':
        st.markdown(f"**Gestora:** {fund_gestora}")

    # ---- DNA Card: Fund vs Cluster gauges ----
    fund_cluster_label = fund.get('cluster_label', '')
    _dna_merged = results_df.merge(features_df, on='cnpj_clean', how='inner')
    _cluster_peers = _dna_merged[_dna_merged['cluster_label'] == fund_cluster_label]

    _dna_metrics = [
        ("Correlacao", 'corr_ibov', 0, 1, ".2f"),
        ("Beta", 'beta_ibov', 0, 1.5, ".2f"),
        ("Volatilidade", 'ann_volatility', 0, 0.50, ".1%"),
        ("Sharpe", 'sharpe_approx', 0, 2.5, ".2f"),
        ("Tracking Error", 'tracking_error', 0, 0.30, ".1%"),
        ("Small Cap Tilt", 'smll_affinity', -0.15, 0.10, ".3f"),
    ]

    _dna_rows_html = ""
    _dna_summary_parts = []
    for _dm_label, _dm_col, _dm_lo, _dm_hi, _dm_fmt in _dna_metrics:
        _fund_val = fund.get(_dm_col, np.nan)
        _cluster_med = _cluster_peers[_dm_col].median() if _dm_col in _cluster_peers.columns else np.nan

        # Percentile within cluster
        _pctile = np.nan
        if pd.notna(_fund_val) and _dm_col in _cluster_peers.columns:
            _peer_vals = _cluster_peers[_dm_col].dropna()
            if len(_peer_vals) > 0:
                _pctile = float((_peer_vals < _fund_val).mean() * 100)

        # Gauge bar percentage (fund value)
        _fund_pct = 0
        if pd.notna(_fund_val):
            _range = _dm_hi - _dm_lo
            _fund_pct = max(0, min(100, (_fund_val - _dm_lo) / _range * 100)) if _range != 0 else 0

        # Cluster median marker position
        _cluster_pct = 0
        if pd.notna(_cluster_med):
            _range = _dm_hi - _dm_lo
            _cluster_pct = max(0, min(100, (_cluster_med - _dm_lo) / _range * 100)) if _range != 0 else 0

        # Format display values
        if pd.notna(_fund_val):
            if '%' in _dm_fmt:
                _fv_str = f"{_fund_val:{_dm_fmt}}"
            else:
                _fv_str = f"{_fund_val:{_dm_fmt}}"
        else:
            _fv_str = "–"

        if pd.notna(_cluster_med):
            if '%' in _dm_fmt:
                _cm_str = f"{_cluster_med:{_dm_fmt}}"
            else:
                _cm_str = f"{_cluster_med:{_dm_fmt}}"
        else:
            _cm_str = "–"

        _pctile_str = f"{_pctile:.0f}%" if pd.notna(_pctile) else "–"

        # Build summary for highest-percentile metric
        if pd.notna(_pctile):
            _dna_summary_parts.append((_dm_label, _pctile, _dm_col))

        _dna_rows_html += (
            f'<div style="display:flex;align-items:center;margin-bottom:10px;">'
            f'<span style="width:120px;font-size:0.80rem;color:{TEXT_MUTED};flex-shrink:0;">{_dm_label}</span>'
            f'<div style="flex:1;position:relative;background:rgba(255,255,255,0.05);border-radius:4px;height:10px;margin:0 10px;">'
            f'<div style="background:{TAG_LARANJA};height:100%;width:{_fund_pct:.1f}%;border-radius:4px;opacity:0.85;"></div>'
            f'<div style="position:absolute;top:-2px;left:{_cluster_pct:.1f}%;width:2px;height:14px;'
            f'background:{TAG_BRANCO};border-radius:1px;opacity:0.9;" title="Mediana cluster: {_cm_str}"></div>'
            f'</div>'
            f'<span style="width:55px;text-align:right;font-size:0.80rem;color:{TAG_OFFWHITE};flex-shrink:0;">{_fv_str}</span>'
            f'<span style="width:45px;text-align:right;font-size:0.72rem;color:{TEXT_MUTED};flex-shrink:0;">P{_pctile_str}</span>'
            f'</div>'
        )

    # Pick the most notable percentile for the summary note
    _dna_note_html = ""
    if _dna_summary_parts:
        _dna_summary_parts.sort(key=lambda x: abs(x[1] - 50), reverse=True)
        _top_label, _top_pctile, _ = _dna_summary_parts[0]
        _dna_note_html = (
            f'<div style="margin-top:12px;font-size:0.78rem;color:{TEXT_MUTED};font-style:italic;">'
            f'Este fundo esta no percentil {_top_pctile:.0f} em {_top_label.lower()} dentro do grupo '
            f'<span style="color:{TAG_OFFWHITE};font-weight:600;">{fund_cluster_label}</span>.</div>'
        )

    # Legend for the markers
    _dna_legend_html = (
        f'<div style="display:flex;gap:16px;margin-top:10px;font-size:0.72rem;color:{TEXT_MUTED};">'
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{TAG_LARANJA};'
        f'border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Fundo</span>'
        f'<span><span style="display:inline-block;width:2px;height:12px;background:{TAG_BRANCO};'
        f'border-radius:1px;vertical-align:middle;margin-right:4px;"></span>Mediana do cluster</span>'
        f'<span>P = Percentil no cluster</span>'
        f'</div>'
    )

    _fund_card_html = (
        f'<div style="background:{TAG_BG_CARD};border-radius:12px;padding:24px 28px;border-left:4px solid {TAG_LARANJA};margin:16px 0 20px 0;">'
        f'<div style="color:{TAG_OFFWHITE};font-size:1.05rem;font-weight:600;margin-bottom:4px;">DNA do Fundo</div>'
        f'<div style="color:{TEXT_MUTED};font-size:0.78rem;margin-bottom:16px;">vs cluster <b style="color:{TAG_OFFWHITE};">{fund_cluster_label}</b> ({len(_cluster_peers)} fundos)</div>'
        f'{_dna_rows_html}'
        f'{_dna_legend_html}'
        f'{_dna_note_html}'
        f'</div>'
    )
    st.markdown(_fund_card_html, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    # Features do fundo
    with col1:
        st.subheader("Features")
        feat_data = []
        for feat in FEATURE_COLS:
            if feat in fund.index:
                val = fund[feat]
                if pd.notna(val):
                    if 'volatility' in feat or 'vol' in feat.lower() or feat in ['ann_return', 'alpha_ibov']:
                        formatted = f"{val:.1%}"
                    elif 'capture' in feat:
                        formatted = f"{val:.2f}x"
                    elif feat.startswith('pct_'):
                        formatted = f"{val:.1%}"
                    elif feat == 'hhi_portfolio':
                        formatted = f"{val:.3f}"
                    elif feat == 'turnover':
                        formatted = f"{val:.1%}"
                    elif feat == 'n_positions':
                        formatted = f"{int(val)}"
                    else:
                        formatted = f"{val:.3f}"
                    # Percentil vs todos os fundos
                    pctile = None
                    if feat in features_df.columns:
                        all_vals = features_df[feat].dropna()
                        if len(all_vals) > 0:
                            pctile = float((all_vals < val).mean() * 100)
                    feat_data.append({
                        'Feature': FEATURE_LABELS.get(feat, feat),
                        'Valor': formatted,
                        'Percentil': pctile,
                    })
        feat_df = pd.DataFrame(feat_data)
        st.dataframe(feat_df, width="stretch", hide_index=True,
            column_config={
                "Percentil": st.column_config.ProgressColumn(
                    "Percentil", min_value=0, max_value=100, format="%.0f%%",
                    help="Posicao relativa vs todos os fundos clusterizados"),
            })
        st.caption("Features do fundo + percentil (posicao relativa vs todos os fundos). Percentil 90% = supera 90% dos fundos nessa metrica.")

    # Radar do fundo vs cluster
    with col2:
        st.subheader("Radar de Features")
        radar_feats = [f for f in RADAR_FEATURES if f in merged.columns]
        labels_radar = [FEATURE_LABELS.get(f, f) for f in radar_feats]

        fund_vals = _compute_radar_values(fund, radar_feats, merged)

        # Media do cluster do fundo
        fund_cluster = fund.get('cluster_label', '')
        cluster_data_af = merged[merged['cluster_label'] == fund_cluster]
        cluster_means_dict = {f: cluster_data_af[f].mean() for f in radar_feats}
        cluster_vals = _compute_radar_values(cluster_means_dict, radar_feats, merged)

        fig = _build_radar_chart([
            {'values': fund_vals, 'labels': labels_radar,
             'name': fund['nome'][:30], 'color': TAG_LARANJA, 'fill': True},
            {'values': cluster_vals, 'labels': labels_radar,
             'name': f'Cluster: {fund_cluster}', 'color': '#3B82F6', 'fill': False, 'dash': 'dot'},
            {'values': [0.5] * len(radar_feats), 'labels': labels_radar,
             'name': 'Media Geral', 'color': TEXT_MUTED, 'fill': False, 'dash': 'dot'},
        ])
        st.plotly_chart(fig, width="stretch")
        st.caption("Fundo (laranja) vs media do cluster (azul pontilhado) vs media geral (cinza). Acima de 0.5 = supera a media.")

    # Peers (abaixo do radar, full width)
    st.subheader("Top 5 Peers")
    peers_str = fund.get('nearest_peers', '')
    if peers_str:
        peer_cnpjs = peers_str.split(';')[:5]
        peer_rows = []
        for pc in peer_cnpjs:
            peer_data = merged[merged['cnpj_clean'] == pc]
            if not peer_data.empty:
                p = peer_data.iloc[0]
                peer_rows.append({
                    'Fundo': get_fund_name(pc, hist_df)[:50],
                    'Cluster': p.get('cluster_label', '-'),
                    'Beta': round(p.get('beta_ibov', 0), 2),
                    'Vol': f"{p.get('ann_volatility', 0):.1%}",
                    'Sharpe': round(p.get('sharpe_approx', 0), 3),
                    'Corr IBOV': round(p.get('corr_ibov', 0), 3),
                })
        if peer_rows:
            st.dataframe(pd.DataFrame(peer_rows), width="stretch", hide_index=True)
    else:
        st.info("Peers nao disponiveis")

    # Sector breakdown
    if not sector_df.empty:
        fund_sectors = sector_df[sector_df['cnpj_clean'] == cnpj]
        if not fund_sectors.empty:
            st.subheader("Composicao Setorial")
            sector_sorted = fund_sectors.sort_values('peso', ascending=False)
            fig = go.Figure(go.Bar(
                x=sector_sorted['setor'],
                y=sector_sorted['peso'],
                marker_color=TAG_LARANJA,
                text=[f"{p:.1%}" for p in sector_sorted['peso']],
                textposition='outside',
                textfont=dict(color=TAG_OFFWHITE),
            ))
            fig.update_layout(**_chart_layout("", height=CHART_HEIGHT_SMALL, showlegend=False))
            fig.update_layout(yaxis=dict(tickformat='.0%'))
            st.plotly_chart(fig, width="stretch")
            st.caption("Alocacao setorial da carteira do fundo (dados BLC_4 CVM). Mostra onde o fundo concentra seus investimentos. Setores com peso < 1% foram omitidos.")

    # Retorno acumulado
    st.subheader("Retorno Acumulado")
    # Carregar cotas sob demanda
    peer_cnpjs = [cnpj]
    fund_cotas_all = load_cotas(tuple(peer_cnpjs))
    fund_cotas = fund_cotas_all[fund_cotas_all['cnpj_clean'] == cnpj].sort_values('dt_comptc')
    if not fund_cotas.empty:
        fund_cotas = fund_cotas.set_index('dt_comptc')['vl_quota']
        fund_cotas = fund_cotas[~fund_cotas.index.duplicated(keep='last')]
        fund_ret_cum = (fund_cotas / fund_cotas.iloc[0] - 1) * 100

        # Ibovespa
        ibov = ibov_df.set_index('dt')['value']
        ibov_start = ibov.loc[ibov.index >= fund_ret_cum.index[0]].iloc[0] if len(ibov) > 0 else 1
        ibov_cum = (ibov / ibov_start - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fund_ret_cum.index, y=fund_ret_cum.values,
            name=fund['nome'][:30], line=dict(color=TAG_LARANJA, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ibov_cum.index, y=ibov_cum.values,
            name='Ibovespa', line=dict(color=TEXT_MUTED, width=1, dash='dot'),
        ))
        fig.update_layout(**_chart_layout("", height=CHART_HEIGHT_MEDIUM))
        fig.update_layout(yaxis=dict(ticksuffix='%'))
        st.plotly_chart(fig, width="stretch")
        st.caption("Retorno acumulado do fundo vs Ibovespa desde a primeira cota disponivel. Permite avaliar se o fundo superou o benchmark no periodo e em quais momentos houve divergencia.")
    else:
        st.info("Cotas diarias nao disponiveis")


# ============================================================
# PAGINA 5: COMPARACAO
# ============================================================
def page_comparacao(features_df, results_df, cotas_df, ibov_df, hist_df):
    st.header("⚖️ Comparacao de Fundos")

    merged = results_df.merge(features_df, on='cnpj_clean', how='inner')
    merged['nome'] = merged['cnpj_clean'].apply(lambda c: get_fund_name(c, hist_df))

    # Adicionar peer_group
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        pg_map = dict(zip(hist_df['cnpj_clean'], hist_df['peer_group']))
        merged['Classificacao'] = merged['cnpj_clean'].map(pg_map).fillna('-')

    # Multi-select
    all_options = [f"{row['nome']} ({row['cnpj_clean']})" for _, row in merged.iterrows()]
    selected = st.multiselect(
        "Selecione 2-5 fundos para comparar",
        all_options,
        max_selections=5,
    )

    if len(selected) < 2:
        st.info("Selecione pelo menos 2 fundos para comparar")
        return

    # Parse CNPJs
    selected_cnpjs = []
    for s in selected:
        cnpj = s.split('(')[-1].rstrip(')')
        selected_cnpjs.append(cnpj)

    sel_data = merged[merged['cnpj_clean'].isin(selected_cnpjs)]

    # Resumo com classificacao e cluster
    if 'Classificacao' in sel_data.columns:
        summary_rows = []
        for _, row in sel_data.iterrows():
            summary_rows.append({
                'Fundo': row['nome'][:40],
                'Classificacao': row.get('Classificacao', '-'),
                'Cluster': row.get('cluster_label', '-'),
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    # Radar comparativo
    st.subheader("Radar Comparativo")
    radar_feats = [f for f in RADAR_FEATURES if f in sel_data.columns]
    labels_radar = [FEATURE_LABELS.get(f, f) for f in radar_feats]

    radar_traces = []
    for i, (_, row) in enumerate(sel_data.iterrows()):
        color = TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)]
        fund_vals = _compute_radar_values(row, radar_feats, merged)
        radar_traces.append({
            'values': fund_vals, 'labels': labels_radar,
            'name': row['nome'][:30], 'color': color,
            'fill': (i == 0),  # fill apenas o primeiro para nao poluir
        })
    # Media geral
    radar_traces.append({
        'values': [0.5] * len(radar_feats), 'labels': labels_radar,
        'name': 'Media Geral', 'color': TEXT_MUTED, 'fill': False, 'dash': 'dot',
    })
    fig = _build_radar_chart(radar_traces)
    st.plotly_chart(fig, width="stretch")
    st.caption("Radar sobrepondo os fundos selecionados. Acima de 0.5 = supera a media geral; abaixo = fica aquem. Permite comparar perfis visualmente.")

    # Features side-by-side
    st.subheader("Features Comparativas")
    compare_feats = ['beta_ibov', 'ann_volatility', 'corr_ibov', 'sharpe_approx',
                     'tracking_error', 'beta_smll', 'corr_smll', 'beta_idiv',
                     'up_capture', 'down_capture', 'smll_affinity', 'idiv_affinity']
    compare_feats = [f for f in compare_feats if f in sel_data.columns]

    compare_df = sel_data.set_index('nome')[compare_feats].T
    compare_df.index = [FEATURE_LABELS.get(f, f) for f in compare_feats]
    # Aplicar heatmap verde-vermelho por linha
    try:
        styled = compare_df.round(3).style.background_gradient(
            cmap='RdYlGn', axis=1
        ).format(precision=3)
        st.dataframe(styled, width="stretch")
    except Exception:
        st.dataframe(compare_df.round(3), width="stretch")
    st.caption("Tabela com heatmap: verde = valor alto, vermelho = valor baixo (relativo entre os fundos selecionados). Compare Beta, Sharpe, Vol e outros indicadores.")

    # Bar chart comparativo
    st.subheader("Comparativo Visual")
    feat_to_compare = st.selectbox("Feature", compare_feats,
                                    format_func=lambda x: FEATURE_LABELS.get(x, x))

    fig = go.Figure()
    for i, (_, row) in enumerate(sel_data.iterrows()):
        color = TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)]
        val = row.get(feat_to_compare, 0)
        fig.add_trace(go.Bar(
            x=[row['nome'][:25]],
            y=[val],
            name=row['nome'][:25],
            marker_color=color,
            text=[f"{val:.3f}"],
            textposition='outside',
            textfont=dict(color=TAG_OFFWHITE),
        ))
    fig.update_layout(**_chart_layout("", height=CHART_HEIGHT_SMALL))
    st.plotly_chart(fig, width="stretch")
    st.caption("Grafico de barras comparando a feature selecionada acima entre os fundos escolhidos. Use o seletor para alternar entre diferentes metricas e ver quem se destaca em cada dimensao.")

    # Retorno acumulado comparativo
    st.subheader("Retorno Acumulado")
    fig = go.Figure()

    # Carregar cotas sob demanda
    cotas_comp = load_cotas(tuple(selected_cnpjs))

    # Encontrar data inicio comum
    min_dates = []
    for cnpj in selected_cnpjs:
        fc = cotas_comp[cotas_comp['cnpj_clean'] == cnpj]
        if not fc.empty:
            min_dates.append(fc['dt_comptc'].min())

    if min_dates:
        common_start = max(min_dates)

        for i, cnpj in enumerate(selected_cnpjs):
            fc = cotas_comp[cotas_comp['cnpj_clean'] == cnpj].sort_values('dt_comptc')
            fc = fc[fc['dt_comptc'] >= common_start]
            if not fc.empty:
                fc = fc.set_index('dt_comptc')['vl_quota']
                fc = fc[~fc.index.duplicated(keep='last')]
                cum = (fc / fc.iloc[0] - 1) * 100
                name = get_fund_name(cnpj, hist_df)[:30]
                color = TAG_CHART_COLORS[i % len(TAG_CHART_COLORS)]
                fig.add_trace(go.Scatter(
                    x=cum.index, y=cum.values,
                    name=name, line=dict(color=color, width=2),
                ))

        # Ibovespa
        ibov = ibov_df.set_index('dt')['value']
        ibov_period = ibov[ibov.index >= common_start]
        if not ibov_period.empty:
            ibov_cum = (ibov_period / ibov_period.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=ibov_cum.index, y=ibov_cum.values,
                name='Ibovespa', line=dict(color=TEXT_MUTED, width=1, dash='dot'),
            ))

        fig.update_layout(**_chart_layout("", height=CHART_HEIGHT_MEDIUM))
        fig.update_layout(yaxis=dict(ticksuffix='%'))
        st.plotly_chart(fig, width="stretch")
        st.caption("Retorno acumulado dos fundos selecionados e Ibovespa, a partir da data inicial comum. Permite visualizar quem superou o benchmark e a consistencia de cada fundo ao longo do tempo.")


# ============================================================
# MAIN
# ============================================================
def _apply_peer_group_filter(results_df, features_df, hist_df, selected_pg):
    """Filtra results_df e features_df pelo peer_group selecionado."""
    if selected_pg == "Todos":
        return results_df, features_df
    if hist_df.empty or 'peer_group' not in hist_df.columns:
        return results_df, features_df
    valid_cnpjs = set(hist_df[hist_df['peer_group'] == selected_pg]['cnpj_clean'])
    results_df = results_df[results_df['cnpj_clean'].isin(valid_cnpjs)]
    features_df = features_df[features_df['cnpj_clean'].isin(valid_cnpjs)]
    return results_df, features_df


def main():
    inject_css()

    # Sidebar com logo TAG
    _logo_b64 = _get_logo_b64()
    if _logo_b64:
        _sidebar_html = (
            '<div style="text-align:center;padding:20px 0 10px 0;">'
            f'<img src="data:image/png;base64,{_logo_b64}" alt="TAG Investimentos"'
            ' style="max-width:180px;margin-bottom:8px;">'
            f'<div style="height:2px;background:linear-gradient(90deg,transparent,{TAG_LARANJA},{TAG_VERMELHO},transparent);'
            'margin:6px auto;width:70%;"></div>'
            f'<p style="color:{TAG_OFFWHITE};font-size:0.95rem;font-weight:600;margin:8px 0 0 0;">'
            'Clusters FIA</p>'
            '</div>'
        )
        st.sidebar.markdown(_sidebar_html, unsafe_allow_html=True)
    else:
        _sidebar_html = (
            '<div style="text-align:center;padding:20px 0;">'
            f'<h2 style="color:{TAG_OFFWHITE};margin:0;">Clusters FIA</h2>'
            f'<p style="color:{TEXT_MUTED};font-size:0.85rem;">TAG Investimentos</p>'
            '</div>'
        )
        st.sidebar.markdown(_sidebar_html, unsafe_allow_html=True)

    _pages = ["📊 Visao Geral", "🗺️ Mapa de Clusters", "🎯 Perfil do Cluster",
              "🔍 Analise de Fundo", "⚖️ Comparacao"]
    pagina_raw = st.sidebar.radio(
        "Navegacao", _pages,
        label_visibility="collapsed",
    )
    # Strip emoji for page matching
    pagina = pagina_raw.split(" ", 1)[1] if " " in pagina_raw else pagina_raw

    # Load data
    features_df, results_df, sector_df, holdings_df, viz_df, cotas_df, ibov_df, hist_df, metadata_df = load_data()

    if features_df.empty or results_df.empty:
        st.error("Dados de clustering nao encontrados. Execute o pipeline primeiro.")
        st.stop()

    # ---- Filtro por Classificacao (peer_group) ----
    pg_options = ["Todos"]
    if not hist_df.empty and 'peer_group' in hist_df.columns:
        clustered_cnpjs = set(results_df['cnpj_clean'])
        hist_clustered = hist_df[hist_df['cnpj_clean'].isin(clustered_cnpjs)]
        pg_list = sorted(hist_clustered['peer_group'].dropna().unique())
        pg_options += pg_list

    st.sidebar.markdown("---")
    selected_pg = st.sidebar.selectbox(
        "🏷️ Filtrar por Classificacao TAG",
        pg_options,
        index=0,
        key="pg_filter",
    )

    # Aplicar filtro
    results_filtered, features_filtered = _apply_peer_group_filter(
        results_df, features_df, hist_df, selected_pg
    )

    if selected_pg != "Todos":
        n_filtered = len(results_filtered)
        st.sidebar.caption(f"{n_filtered} fundos em **{selected_pg}**")

    # Busca rapida
    st.sidebar.markdown("---")
    _quick_search = st.sidebar.text_input(
        "🔎 Busca rapida",
        placeholder="Nome ou CNPJ...",
        key="sidebar_quick_search",
        help="Digite para buscar um fundo. Abre a pagina Analise de Fundo.",
    )
    if _quick_search and len(_quick_search) >= 3:
        st.session_state['_quick_search_term'] = _quick_search

    # Botao limpar cache
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Atualizar Dados"):
        st.cache_data.clear()
        st.rerun()

    # Pages
    if pagina == "Visao Geral":
        page_visao_geral(features_filtered, results_filtered, hist_df)
    elif pagina == "Mapa de Clusters":
        page_mapa_clusters(features_filtered, results_filtered, viz_df, hist_df, metadata_df)
    elif pagina == "Perfil do Cluster":
        page_perfil_cluster(features_filtered, results_filtered, sector_df, hist_df, metadata_df)
    elif pagina == "Analise de Fundo":
        page_analise_fundo(features_df, results_df, sector_df, cotas_df, ibov_df, hist_df, metadata_df)
    elif pagina == "Comparacao":
        page_comparacao(features_df, results_df, cotas_df, ibov_df, hist_df)


if __name__ == "__main__":
    main()
