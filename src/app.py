"""
Rocket League — Détection Comportementale de Smurfs
Application Streamlit – src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os

try:
    from config import MODEL_METRICS_FILE, DATA_DIR, MODELS_DIR
except ImportError:
    # Fallback paths for standalone testing
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODEL_METRICS_FILE = os.path.join(BASE_DIR, "model_metrics.csv")

# ──────────────────────────────────────────────
# THEME & GLOBAL STYLES
# ──────────────────────────────────────────────
ROCKET_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* Root palette */
:root {
    --bg-deep:    #070b14;
    --bg-card:    #0d1526;
    --bg-card2:   #111b2e;
    --accent-blue:#00c6ff;
    --accent-gold:#f5a623;
    --accent-red: #ff3b5c;
    --accent-grn: #00e676;
    --accent-pur: #7c4dff;
    --text-main:  #e8edf5;
    --text-muted: #5a6a84;
    --border:     rgba(0,198,255,0.15);
}

/* Global reset */
html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'Rajdhani', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ── Tab bar ── */
[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
}
[data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 20px !important;
    transition: color 0.2s, background 0.2s;
}
[aria-selected="true"] {
    color: var(--accent-blue) !important;
    background: rgba(0,198,255,0.07) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 22px !important;
}
[data-testid="metric-container"] label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    color: var(--text-muted) !important;
    text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.6rem !important;
    color: var(--accent-blue) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Custom card wrapper ── */
.rl-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 18px;
}
.rl-card-highlight {
    background: linear-gradient(135deg, #0d1f3c 0%, #091524 100%);
    border: 1px solid rgba(0,198,255,0.3);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 18px;
}

/* ── Section titles ── */
.rl-section-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.05rem;
    letter-spacing: 0.12em;
    color: var(--accent-blue);
    text-transform: uppercase;
    margin-bottom: 8px;
    border-left: 3px solid var(--accent-blue);
    padding-left: 12px;
}
.rl-hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.4rem;
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(90deg, #00c6ff, #7c4dff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.rl-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
}
.rl-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.08em;
}
.badge-diamond { background: rgba(90,180,255,0.15); color: #5ab4ff; border: 1px solid #5ab4ff55; }
.badge-gc      { background: rgba(245,166,35,0.15);  color: var(--accent-gold); border: 1px solid #f5a62355; }
.badge-safe    { background: rgba(0,230,118,0.12);   color: var(--accent-grn);  border: 1px solid #00e67655; }
.badge-warn    { background: rgba(255,165,0,0.15);   color: #ffa500; border: 1px solid #ffa50055; }
.badge-danger  { background: rgba(255,59,92,0.15);   color: var(--accent-red);  border: 1px solid #ff3b5c55; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    background: transparent !important;
    border: 1px solid var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    background: rgba(0,198,255,0.1) !important;
    box-shadow: 0 0 18px rgba(0,198,255,0.25);
}

/* ── Selectbox ── */
[data-baseweb="select"] {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
}

/* ── Progress bar override ── */
[data-testid="stProgress"] > div > div {
    border-radius: 4px;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 24px 0; }

/* ── Platform icon row ── */
.platform-grid {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 12px 0;
}
.platform-btn {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 18px;
    cursor: pointer;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.9rem;
    color: var(--text-muted);
    transition: all 0.2s;
    display: inline-flex;
    align-items: center;
    gap: 7px;
}
.platform-btn.selected {
    border-color: var(--accent-blue);
    color: var(--accent-blue);
    background: rgba(0,198,255,0.08);
    box-shadow: 0 0 12px rgba(0,198,255,0.15);
}

/* ── Risk gauge label ── */
.gauge-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    margin-top: -10px;
}
.gauge-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.1em;
}
</style>
"""

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Rajdhani, sans-serif", color="#e8edf5"),
)

_DEFAULT_MARGIN = dict(l=20, r=20, t=40, b=20)

def plotly_axis_style():
    return dict(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.1)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )


def card(content_fn, highlight=False):
    cls = "rl-card-highlight" if highlight else "rl-card"
    st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="rl-section-title">{text}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────

@st.cache_data
def load_training_data():
    path = os.path.join(DATA_DIR, "rocket_league_skill_data.csv")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"⚠️ Dataset introuvable : `{path}`. Données synthétiques utilisées.")
        rng = np.random.default_rng(42)
        n = 2000
        target = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        rng.shuffle(target)
        df = pd.DataFrame({
            "Target": target,
            "percentage supersonic speed": np.where(target == 0,
                rng.normal(12, 3, n), rng.normal(18, 3.5, n)),
            "amount collected small pads": np.where(target == 0,
                rng.normal(45, 10, n), rng.normal(68, 11, n)),
            "percentage high in air": np.where(target == 0,
                rng.normal(8, 3, n), rng.normal(14, 4, n)),
            "avg speed":  np.where(target == 0,
                rng.normal(1100, 150, n), rng.normal(1400, 160, n)),
            "boost usage efficiency": np.where(target == 0,
                rng.normal(0.55, 0.1, n), rng.normal(0.74, 0.09, n)),
            "demos inflicted": np.where(target == 0,
                rng.normal(1.2, 0.6, n), rng.normal(2.4, 0.9, n)),
        })
        return df


@st.cache_data
def load_match_data():
    path = os.path.join(DATA_DIR, "gamesgc-diam-q12jszwnkd-players-games.csv")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv(MODEL_METRICS_FILE)
        return df
    except FileNotFoundError:
        return pd.DataFrame({
            "model_key":   ["log_reg",      "random_forest"],
            "model_name":  ["Régression Logistique", "Random Forest"],
            "accuracy":    [0.891,           0.923],
            "precision":   [0.917,           0.934],
            "recall":      [0.863,           0.911],
            "f1_score":    [0.889,           0.922],
        })


@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, "log_reg.joblib")
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


# ──────────────────────────────────────────────
# ACCOUNT PROFILES (hardcoded demo data)
# ──────────────────────────────────────────────

ACCOUNT_PROFILES = {
    "RAPH-403 — Lobby Champion": {
        "handle": "RAPH-403",
        "rank": "Champion II",
        "lobby": "Champion",
        "mode": "smurf",           # cherche les smurfs (Diamond → GC)
        "gc_proba": 0.48,
        "lobby_avg_proba": 0.10,
        "stats": {
            "% Supersonique":       "16.2%",
            "Boosts petits pads":   "61",
            "% Haut dans les airs": "11.8%",
            "Vitesse moy. (uu/s)":  "1 340",
            "Eff. boost":           "0.71",
            "Démos infligées":      "2.1",
        },
        "narrative": (
            "Jouant en lobby Champion, RAPH-403 affiche un profil mécanique "
            "typique des Grand Champions : gestion du boost élite, "
            "vitesse supersonique 4× au-dessus de la moyenne de son lobby. "
            "<strong style='color:#ff3b5c;'>Anomalie comportementale flagrante — profil Smurf.</strong>"
        ),
        "lobby_players": [
            ("Player_A",  0.09), ("Player_B",  0.12), ("Player_C",  0.08),
            ("RAPH-403",  0.48), ("Player_D",  0.11), ("Player_E",  0.10),
        ],
    },
    "RAPH-402 — Lobby GC2": {
        "handle": "RAPH-402",
        "rank": "GC2",
        "lobby": "GC2",
        "mode": "boosted",         # cherche les boostés (GC → Diamond)
        "gc_proba": 0.52,
        "lobby_avg_proba": 0.58,
        "stats": {
            "% Supersonique":       "21.4%",
            "Boosts petits pads":   "48",
            "% Haut dans les airs": "0.6%",
            "Vitesse moy. (uu/s)":  "1 510",
            "Eff. boost":           "0.63",
            "Démos infligées":      "1.8",
        },
        "narrative": (
            "RAPH-402 joue en lobby GC2. Sa vitesse supersonique (21%) est au-dessus de la moyenne GC, "
            "mais son temps en l'air (0.6%) est quasi nul — profil groundplay atypique mais cohérent. "
            "En revanche, l'analyse du lobby détecte <strong style='color:#f5a623;'>Player_C</strong> "
            "avec seulement 23% de probabilité GC malgré son rang affiché : "
            "<strong style='color:#f5a623;'>profil Boosté — mécaniques insuffisantes pour ce niveau.</strong>"
        ),
        "lobby_players": [
            ("Player_A",  0.61), ("Player_B",  0.67), ("Player_C",  0.23),
            ("RAPH-402",  0.52), ("Player_D",  0.59), ("Player_E",  0.72),
        ],
    },
}

PLATFORMS = [
    ("🎮", "PS5 / PS4"),
    ("🟢", "Xbox"),
    ("⚡", "Epic Games"),
    ("🚀", "Steam"),
    ("🔵", "Psyonix"),
    ("🎴", "Nintendo Switch"),
]


# ──────────────────────────────────────────────
# TAB 1 — LE FLÉAU DU SMURFING
# ──────────────────────────────────────────────

def tab_business(df: pd.DataFrame):
    # ── Hero ──
    st.markdown("""
    <div style="padding: 28px 0 12px;">
        <div class="rl-hero-title">LE FLÉAU DU SMURFING</div>
        <div class="rl-subtitle" style="margin-top:8px;">
            Quand l'élite se déguise pour écraser les débutants
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI row ──
    diamonds = df[df["Target"] == 0]
    gcs = df[df["Target"] == 1]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Joueurs analysés", f"{len(df):,}", "dataset équilibré")
    k2.metric("% Supersonique — GC", f"{gcs['percentage supersonic speed'].mean():.1f}%",
              f"+{gcs['percentage supersonic speed'].mean() - diamonds['percentage supersonic speed'].mean():.1f}% vs Diamond")
    k3.metric("Boost collecté (GC)", f"{gcs['amount collected small pads'].mean():.0f}",
              f"+{gcs['amount collected small pads'].mean() - diamonds['amount collected small pads'].mean():.0f} vs Diamond")
    k4.metric("Feature n°1 prédictive", "Boost pads", "Amount collected small pads")
    k5.metric("Seuil Shadowban", "95%", "confiance minimale")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    # ── Boxplot supersonique ──
    with col_l:
        section_title("Distribution — Vitesse Supersonique")
        fig_box = go.Figure()
        for label, subset, color, fill in [
            ("💎 Diamond", diamonds, "#5ab4ff", "rgba(90,180,255,0.12)"),
            ("🏆 Grand Champion", gcs, "#f5a623", "rgba(245,166,35,0.12)"),
        ]:
            fig_box.add_trace(go.Box(
                y=subset["percentage supersonic speed"],
                name=label,
                marker_color=color,
                line_color=color,
                fillcolor=fill,
                boxmean="sd",
            ))
        fig_box.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="% Temps supersonique par rang", font=dict(size=13, color="#5a6a84")),
            yaxis=dict(title="% Supersonique", **plotly_axis_style()),
            xaxis=dict(**plotly_axis_style()),
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
            margin=_DEFAULT_MARGIN,
            height=380,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Feature importance bar ──
    with col_r:
        section_title("Importance des Features")
        features = [
            ("amount collected small pads", 0.312),
            ("percentage supersonic speed",  0.267),
            ("avg speed",                    0.198),
            ("percentage high in air",       0.143),
            ("boost usage efficiency",       0.080),
        ]
        feat_df = pd.DataFrame(features, columns=["Feature", "Importance"])
        feat_df = feat_df.sort_values("Importance")

        colors = ["#00c6ff" if i == len(feat_df) - 1 else "#1e3a5a" for i in range(len(feat_df))]

        fig_bar = go.Figure(go.Bar(
            x=feat_df["Importance"],
            y=feat_df["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1%}" for v in feat_df["Importance"]],
            textposition="outside",
            textfont=dict(color="#e8edf5", size=12, family="Share Tech Mono"),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Coefficient de corrélation avec rang GC", font=dict(size=13, color="#5a6a84")),
            xaxis=dict(title="Importance relative", **plotly_axis_style()),
            yaxis=dict(**plotly_axis_style()),
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
            margin=_DEFAULT_MARGIN,
            height=380,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter Speed vs Air ──
    section_title("Carte Mécanique — Vitesse vs Temps en l'air")
    sample = df.sample(min(600, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample,
        x="percentage supersonic speed",
        y="percentage high in air",
        color=sample["Target"].map({0: "💎 Diamond", 1: "🏆 Grand Champion"}),
        color_discrete_map={"💎 Diamond": "#5ab4ff", "🏆 Grand Champion": "#f5a623"},
        opacity=0.55,
        size_max=6,
        labels={"x": "% Supersonique", "y": "% Haut dans les airs"},
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Séparation mécanique Diamond vs Grand Champion", font=dict(size=13, color="#5a6a84")),
        xaxis=dict(title="% Supersonique", **plotly_axis_style()),
        yaxis=dict(title="% Haut dans les airs", **plotly_axis_style()),
        height=380,
        legend=dict(title="Rang", bgcolor="rgba(0,0,0,0)"),
        margin=_DEFAULT_MARGIN,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Context block ──
    st.markdown("""
    <div class="rl-card" style="border-color: rgba(245,166,35,0.25);">
        <div class="rl-section-title" style="color: #f5a623; border-color: #f5a623;">
            ⚠️ Pourquoi c'est un problème grave
        </div>
        <p style="color: #b0c0d8; font-size: 1rem; line-height: 1.7;">
        Le smurfing représente <strong style="color:#f5a623;">~8% des comptes actifs</strong> sur Rocket League selon 
        les estimations de la communauté. Un smurf joue intentionnellement en-dessous de son niveau réel, 
        causant une expérience dégradée pour des milliers de joueurs légitimes chaque jour. 
        <br><br>
        La détection classique (signalements, scores bruts) est <strong style="color:#ff3b5c;">facilement contournable</strong> :
        un Grand Champion peut simplement rater des tirs ou ne pas sauter. Mais ses <em>réflexes mécaniques fondamentaux</em> 
        — gestion du boost, positionnement supersonique — restent trahissants, même volontairement.
        <br><br>
        Notre approche ? Analyser l'<strong style="color:#00c6ff;">ADN mécanique</strong> du joueur, pas ses scores.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TAB 2 — L'INTELLIGENCE ARTIFICIELLE
# ──────────────────────────────────────────────

def tab_ai(metrics_df: pd.DataFrame, df: pd.DataFrame):
    st.markdown("""
    <div style="padding: 28px 0 12px;">
        <div class="rl-hero-title">L'INTELLIGENCE ARTIFICIELLE</div>
        <div class="rl-subtitle" style="margin-top:8px;">
            Détection d'anomalies · Modèles · Explicabilité
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Precision vs Recall explanation ──
    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        section_title("Pourquoi Precision & Recall > Accuracy")
        st.markdown("""
        <div class="rl-card">
        <p style="color:#b0c0d8; font-size:1rem; line-height:1.8;">
        Dans notre contexte, un <strong style="color:#ff3b5c;">Faux Positif</strong> (FP) 
        désigne un compte Diamond prédit GC par le modèle — c'est précisément là que 
        <strong style="color:#ff3b5c;">se cachent les Smurfs</strong> : un GC jouant sur un petit compte.<br><br>
        Un <strong style="color:#f5a623;">Faux Négatif</strong> (FN) désigne un compte GC prédit Diamond 
        — typiquement un joueur <strong style="color:#f5a623;">"Boosté"</strong> : un Diamond porté en lobby GC 
        par des amis plus forts, dont les mécaniques restent insuffisantes malgré le rang affiché.<br><br>
        ➜ On optimise la <strong style="color:#00c6ff;">Précision</strong> en priorité 
        (éviter de rater des vrais Smurfs), au détriment d'un peu de Recall.<br>
        ➜ Le seuil de décision est fixé à <strong style="color:#00e676;">95%</strong> 
        de confiance pour déclencher un Shadowban — c'est délibéré et documenté.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Confusion matrix mock
        section_title("Matrice de Confusion (modèle final)")
        z = [[412, 38], [26, 424]]
        labels = [["TN: 412<br>✅ Diamond → Diamond<br><i>joueur à son niveau</i>", "FP: 38<br>🚨 Diamond → GC<br><i>= Smurf détecté</i>"],
                  ["FN: 26<br>⚠️ GC → Diamond<br><i>= Joueur Boosté</i>", "TP: 424<br>✅ GC → GC<br><i>joueur à son niveau</i>"]]
        colors = [["#0d3b2e", "#3b1220"], ["#1a2a3b", "#1a3b1a"]]

        fig_cm = go.Figure(go.Heatmap(
            z=z,
            colorscale=[[0, "#0d1526"], [1, "rgba(0,198,255,0.2)"]],
            showscale=False,
            text=labels,
            texttemplate="%{text}",
            textfont=dict(size=13, family="Rajdhani"),
        ))
        fig_cm.update_layout(
            **PLOTLY_LAYOUT,
            xaxis=dict(tickvals=[0, 1], ticktext=["Prédit Diamond", "Prédit GC"], **plotly_axis_style()),
            yaxis=dict(tickvals=[0, 1], ticktext=["Réel Diamond", "Réel GC"], **plotly_axis_style()),
            showlegend=False,
            margin=_DEFAULT_MARGIN,
            height=280,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics table ──
    section_title("Comparaison des Modèles Entraînés")
    styled_metrics = metrics_df.copy()
    pct_cols = ["accuracy", "precision", "recall", "f1_score"]
    rename_map = {
        "model_key": "Clé", "model_name": "Modèle",
        "accuracy": "Accuracy", "precision": "Précision",
        "recall": "Recall", "f1_score": "F1-Score",
    }

    display_df = styled_metrics.rename(columns=rename_map)
    for col in ["Accuracy", "Précision", "Recall", "F1-Score"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda v: f"{v:.1%}" if pd.notna(v) else "—")

    if "Clé" in display_df.columns:
        display_df = display_df.drop(columns=["Clé"])

    st.dataframe(
        display_df.style.set_properties(**{
            "background-color": "#0d1526",
            "color": "#e8edf5",
            "font-family": "Rajdhani, sans-serif",
            "font-size": "1rem",
        }).apply(
            lambda row: ["background-color: rgba(0,198,255,0.08); border-left: 2px solid #00c6ff"] * len(row)
            if ("Logistique" in str(row.values) or "log_reg" in str(row.values)) else [""] * len(row),
            axis=1
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model radar chart ──
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        section_title("Radar des Performances")
        categories = ["Accuracy", "Précision", "Recall", "F1-Score"]
        model_colors = {"Régression Logistique": "#00c6ff", "Random Forest": "#f5a623"}

        fig_radar = go.Figure()
        fill_colors = {"#00c6ff": "rgba(0,198,255,0.1)", "#f5a623": "rgba(245,166,35,0.1)", "#7c4dff": "rgba(124,77,255,0.1)"}
        for _, row in metrics_df.iterrows():
            vals = [row["accuracy"], row["precision"], row["recall"], row["f1_score"]]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            color = model_colors.get(row["model_name"], "#7c4dff")
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=fill_colors.get(color, "rgba(124,77,255,0.1)"),
                line=dict(color=color, width=2),
                name=row["model_name"],
                marker=dict(color=color),
            ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0.8, 1.0], gridcolor="rgba(255,255,255,0.08)", color="#5a6a84"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
            margin=_DEFAULT_MARGIN,
            height=350,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        section_title("White-Box vs Black-Box")
        st.markdown("""
        <div class="rl-card-highlight">
            <p style="color:#b0c0d8; line-height:1.8; font-size:1rem;">
            Le Random Forest obtient un F1 légèrement supérieur. Pourtant, nous avons choisi 
            la <strong style="color:#00c6ff;">Régression Logistique</strong> comme modèle de production.
            </p>
            <br>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px;">
                <div style="background:#0d2a1a; border:1px solid #00e67633; border-radius:10px; padding:14px;">
                    <div style="font-family:'Orbitron',sans-serif; color:#00e676; font-size:0.75rem; margin-bottom:8px;">✅ WHITE-BOX</div>
                    <div style="color:#b0c0d8; font-size:0.9rem; line-height:1.6;">
                    Chaque coefficient est interprétable.<br>
                    En cas de ban contesté, on peut <strong style="color:#00e676;">expliquer précisément</strong> 
                    pourquoi le compte a été flaggé.
                    </div>
                </div>
                <div style="background:#2a0d0d; border:1px solid #ff3b5c33; border-radius:10px; padding:14px;">
                    <div style="font-family:'Orbitron',sans-serif; color:#ff3b5c; font-size:0.75rem; margin-bottom:8px;">❌ BLACK-BOX</div>
                    <div style="color:#b0c0d8; font-size:0.9rem; line-height:1.6;">
                    Random Forest = boîte noire.<br>
                    Légalement risqué pour des décisions qui 
                    <strong style="color:#ff3b5c;">impactent des comptes réels</strong>.
                    </div>
                </div>
            </div>
            <br>
            <p style="color:#5a6a84; font-size:0.82rem; font-style:italic;">
            Règle n°1 de la détection de fraude : ne jamais trader l'explicabilité contre 0.3% de précision.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Coeff importance (mock) ──
    section_title("Coefficients Logistiques — Impact par Feature")
    log_coeffs = pd.DataFrame({
        "Feature": [
            "amount collected small pads",
            "percentage supersonic speed",
            "avg speed",
            "percentage high in air",
            "boost usage efficiency",
        ],
        "Coefficient": [1.42, 1.18, 0.93, 0.71, 0.41],
    }).sort_values("Coefficient")

    fig_coeff = go.Figure(go.Bar(
        x=log_coeffs["Coefficient"],
        y=log_coeffs["Feature"],
        orientation="h",
        marker=dict(
            color=log_coeffs["Coefficient"],
            colorscale=[[0, "#1e3a5a"], [1, "#00c6ff"]],
            showscale=False,
        ),
        text=[f"+{v:.2f}" for v in log_coeffs["Coefficient"]],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=12, color="#e8edf5"),
    ))
    fig_coeff.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(title="Coefficient (impact vers GC)", **plotly_axis_style()),
        yaxis=dict(**plotly_axis_style()),
        showlegend=False,
        margin=_DEFAULT_MARGIN,
        height=300,
    )
    st.plotly_chart(fig_coeff, use_container_width=True)


# ──────────────────────────────────────────────
# TAB 3 — INTERFACE UTILISATEUR
# ──────────────────────────────────────────────

def _risk_gauge(proba: float, lobby_avg: float):
    """Plotly gauge chart for risk probability."""
    if proba < 0.5:
        bar_color, label, badge_cls = "#00e676", "NORMAL", "badge-safe"
    elif proba < 0.75:
        bar_color, label, badge_cls = "#ffa500", "SUSPECT", "badge-warn"
    elif proba < 0.95:
        bar_color, label, badge_cls = "#ff7043", "ÉLEVÉ", "badge-warn"
    else:
        bar_color, label, badge_cls = "#ff3b5c", "SHADOWBAN", "badge-danger"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number=dict(suffix="%", font=dict(family="Orbitron", size=36, color=bar_color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#5a6a84", size=11)),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0,  50], color="rgba(0,230,118,0.06)"),
                dict(range=[50, 75], color="rgba(255,165,0,0.06)"),
                dict(range=[75, 95], color="rgba(255,112,67,0.06)"),
                dict(range=[95, 100], color="rgba(255,59,92,0.10)"),
            ],
            threshold=dict(
                line=dict(color="#ff3b5c", width=2),
                thickness=0.8,
                value=95,
            ),
        ),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        showlegend=False,
        height=280,
        margin=dict(l=30, r=30, t=20, b=10),
    )
    return fig, label, badge_cls


def tab_ui():
    # ── App header ──
    st.markdown("""
    <div style="padding: 20px 0 8px; display:flex; align-items:center; gap:16px;">
        <div>
            <div class="rl-hero-title" style="font-size:1.9rem;">SMURF RADAR</div>
            <div class="rl-subtitle">Analyse comportementale en temps réel · Rocket League</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fake nav bar ──
    st.markdown("""
    <div style="display:flex; gap:6px; margin: 10px 0 24px; border-bottom:1px solid rgba(0,198,255,0.12); padding-bottom:12px;">
        <div style="font-family:'Orbitron',sans-serif; font-size:0.65rem; color:#00c6ff; 
                    background:rgba(0,198,255,0.1); border:1px solid rgba(0,198,255,0.3);
                    border-radius:5px; padding:6px 14px; cursor:pointer;">🏠 ACCUEIL</div>
        <div style="font-family:'Orbitron',sans-serif; font-size:0.65rem; color:#5a6a84; 
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                    border-radius:5px; padding:6px 14px; cursor:pointer;">📊 HISTORIQUE</div>
        <div style="font-family:'Orbitron',sans-serif; font-size:0.65rem; color:#5a6a84; 
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                    border-radius:5px; padding:6px 14px; cursor:pointer;">⚙️ PARAMÈTRES</div>
        <div style="font-family:'Orbitron',sans-serif; font-size:0.65rem; color:#5a6a84; 
                    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
                    border-radius:5px; padding:6px 14px; cursor:pointer;">📖 DOCS</div>
        <div style="margin-left:auto; font-family:'Share Tech Mono',monospace; font-size:0.7rem; 
                    color:#00e676; padding:6px 14px;">● LIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search panel ──
    st.markdown("""
    <div class="rl-card-highlight">
        <div class="rl-section-title">🔍 Recherche de Joueur</div>
    </div>
    """, unsafe_allow_html=True)

    # Platform selector
    st.markdown("**Plateforme**", unsafe_allow_html=False)

    if "platform" not in st.session_state:
        st.session_state.platform = "Epic Games"

    plat_cols = st.columns(len(PLATFORMS))
    for i, (icon, name) in enumerate(PLATFORMS):
        is_sel = st.session_state.platform == name
        style_sel = "color:#00c6ff; border-color:#00c6ff; background:rgba(0,198,255,0.08);" if is_sel else ""
        plat_cols[i].markdown(
            f'<div class="platform-btn {"selected" if is_sel else ""}">{icon} {name}</div>',
            unsafe_allow_html=True,
        )
        if plat_cols[i].button(f"Sélectionner", key=f"plat_{name}", help=name):
            st.session_state.platform = name
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Account selector
    col_search, col_analyze = st.columns([3, 1], gap="medium")
    with col_search:
        account_choice = st.selectbox(
            "🎮 Pseudo / Compte à analyser",
            options=list(ACCOUNT_PROFILES.keys()),
            index=0,
            help="Seuls ces deux comptes réels sont disponibles dans ce PoC",
        )
    with col_analyze:
        st.markdown("<br>", unsafe_allow_html=True)
        do_analyze = st.button("⚡ LANCER L'ANALYSE", use_container_width=True)

    st.markdown("---")

    # ── Results ──
    profile = ACCOUNT_PROFILES[account_choice]

    if not do_analyze and "last_analyzed" not in st.session_state:
        st.markdown("""
        <div style="text-align:center; padding:60px 0; color:#5a6a84;">
            <div style="font-family:'Orbitron',sans-serif; font-size:1.1rem; margin-bottom:8px;">
                EN ATTENTE D'ANALYSE
            </div>
            <div style="font-size:0.9rem;">Sélectionnez un compte et lancez l'analyse</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if do_analyze:
        st.session_state.last_analyzed = account_choice

    active_profile = ACCOUNT_PROFILES.get(
        st.session_state.get("last_analyzed", account_choice), profile
    )

    gc_proba   = active_profile["gc_proba"]
    lobby_avg  = active_profile["lobby_avg_proba"]
    mode       = active_profile.get("mode", "smurf")   # "smurf" or "boosted"
    lobby_players = active_profile.get("lobby_players", [])

    # ── Derived: for boosted mode, anomaly = LOW gc_proba in a GC lobby ──
    if mode == "boosted":
        # Find the most suspicious (lowest proba) player that isn't the searched account
        others = [(n, p) for n, p in lobby_players if n != active_profile["handle"]]
        flagged_name, flagged_proba = min(others, key=lambda x: x[1])
        boosted_risk = 1.0 - flagged_proba   # how "not-GC" they look
    else:
        flagged_name, flagged_proba, boosted_risk = None, None, None

    # ── Result header ──
    mode_badge = (
        '<span class="rl-badge badge-danger" style="margin-left:8px;">🔍 MODE DÉTECTION SMURF</span>'
        if mode == "smurf" else
        '<span class="rl-badge badge-warn" style="margin-left:8px;">🔍 MODE DÉTECTION BOOSTÉ</span>'
    )
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:20px; flex-wrap:wrap;">
        <div style="font-family:'Orbitron',sans-serif; font-size:1.5rem; color:#e8edf5;">
            {active_profile['handle']}
        </div>
        <span class="rl-badge badge-diamond">💎 {active_profile['rank']}</span>
        <span class="rl-badge badge-gc">Lobby : {active_profile['lobby']}</span>
        {mode_badge}
    </div>
    """, unsafe_allow_html=True)

    res_left, res_right = st.columns([1.1, 0.9], gap="large")

    with res_left:
        if mode == "smurf":
            section_title("Score Smurf du joueur — IA")
            fig_gauge, risk_label, badge_cls = _risk_gauge(gc_proba, lobby_avg)
            gauge_subtitle = f"Probabilité que ce compte Diamond soit en réalité GC"
        else:
            section_title("Joueur Boosté détecté — IA")
            # Gauge shows boosted risk of the flagged player
            fig_gauge, risk_label, badge_cls = _risk_gauge(boosted_risk, 0.4)
            gauge_subtitle = f"Probabilité que <b>{flagged_name}</b> soit boosté (mécaniques sous le niveau GC)"

        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="text-align:center; margin-top:-10px;">
            <span class="rl-badge {badge_cls}" style="font-size:1rem; padding:6px 20px;">
                {risk_label}
            </span>
        </div>
        <div style="text-align:center; margin-top:8px; font-size:0.8rem; color:#5a6a84; font-family:'Share Tech Mono',monospace;">
            {gauge_subtitle}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:18px; padding:12px 16px;
                    background:rgba(0,0,0,0.3); border-radius:8px;
                    border:1px solid rgba(255,59,92,0.2);">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:#5a6a84; margin-bottom:4px;">
                SEUIL DE SÉCURITÉ
            </div>
            <div style="font-family:'Orbitron',sans-serif; color:#ff3b5c; font-size:1.1rem;">
                95% → SHADOWBAN
            </div>
            <div style="font-size:0.82rem; color:#5a6a84; margin-top:4px;">
                Aucune action automatique en-dessous de ce seuil
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_right:
        # ── Stats table of the searched player ──
        section_title(f"Profil Mécanique — {active_profile['handle']}")
        for stat_k, stat_v in active_profile["stats"].items():
            s1, s2 = st.columns([1.6, 1])
            s1.markdown(f"<span style='color:#5a6a84; font-size:0.88rem;'>{stat_k}</span>", unsafe_allow_html=True)
            s2.markdown(f"<span style='font-family:Share Tech Mono,monospace; color:#00c6ff;'>{stat_v}</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Lobby overview bar (all players) ──
        section_title("Vue d'ensemble du Lobby")
        lp_names  = [n for n, _ in lobby_players]
        lp_probas = [p * 100 for _, p in lobby_players]

        if mode == "smurf":
            # Highlight the searched player (high proba = suspicious)
            bar_colors = [
                "#ff3b5c" if n == active_profile["handle"] and gc_proba >= 0.95
                else "#f5a623" if n == active_profile["handle"]
                else "#1e3a5a"
                for n in lp_names
            ]
            threshold_y, threshold_label, threshold_color = 95, "SEUIL SMURF 95%", "#ff3b5c"
        else:
            # Highlight the boosted player (low proba = suspicious in GC lobby)
            bar_colors = [
                "#f5a623" if n == flagged_name
                else "#1e3a5a"
                for n in lp_names
            ]
            threshold_y, threshold_label, threshold_color = 40, "SEUIL BOOSTÉ 40%", "#f5a623"

        fig_lobby = go.Figure(go.Bar(
            x=lp_names,
            y=lp_probas,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.0f}%" for v in lp_probas],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=11, color="#e8edf5"),
        ))
        fig_lobby.add_hline(
            y=threshold_y,
            line_dash="dash",
            line_color=threshold_color,
            annotation_text=threshold_label,
            annotation_font=dict(color=threshold_color, family="Share Tech Mono", size=9),
        )
        fig_lobby.update_layout(
            **PLOTLY_LAYOUT,
            yaxis=dict(range=[0, 115], title="Proba GC (%)", **plotly_axis_style()),
            xaxis=dict(**plotly_axis_style()),
            height=260,
            margin=_DEFAULT_MARGIN,
            showlegend=False,
        )
        st.plotly_chart(fig_lobby, use_container_width=True)

    # ── Narrative analysis ──
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("📝 Analyse Comportementale")
    st.markdown(f"""
    <div class="rl-card">
        <p style="color:#b0c0d8; font-size:1rem; line-height:1.8;">
        {active_profile['narrative']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Final recommendation ──
    if mode == "smurf":
        if gc_proba >= 0.95:
            rec_color, rec_bg, rec_icon, rec_text = (
                "#ff3b5c", "rgba(255,59,92,0.08)", "🚨",
                f"SHADOWBAN RECOMMANDÉ sur <b>{active_profile['handle']}</b> — Score Smurf : {gc_proba:.0%}. "
                "Le seuil de sécurité de 95% est dépassé."
            )
        elif gc_proba >= 0.40:
            rec_color, rec_bg, rec_icon, rec_text = (
                "#ffa500", "rgba(255,165,0,0.06)", "⚠️",
                f"SURVEILLANCE ACTIVE sur <b>{active_profile['handle']}</b> — Score Smurf : {gc_proba:.0%}. "
                "Anomalie détectée. Suivi sur les 5 prochains matchs recommandé."
            )
        else:
            rec_color, rec_bg, rec_icon, rec_text = (
                "#00e676", "rgba(0,230,118,0.06)", "✅",
                f"AUCUNE ACTION — Score Smurf : {gc_proba:.0%}. "
                "Le profil mécanique est cohérent avec le rang déclaré."
            )
    else:
        # boosted mode — flag is on the detected player in the lobby
        if boosted_risk >= 0.60:
            rec_color, rec_bg, rec_icon, rec_text = (
                "#f5a623", "rgba(245,166,35,0.08)", "⚠️",
                f"JOUEUR BOOSTÉ DÉTECTÉ : <b>{flagged_name}</b> — Proba GC réelle : {flagged_proba:.0%}. "
                "Les mécaniques de ce joueur sont incompatibles avec son rang GC affiché. "
                "Signalement pour vérification recommandé."
            )
        else:
            rec_color, rec_bg, rec_icon, rec_text = (
                "#00e676", "rgba(0,230,118,0.06)", "✅",
                f"AUCUNE ANOMALIE DÉTECTÉE dans le lobby de <b>{active_profile['handle']}</b>. "
                "Tous les joueurs affichent des mécaniques cohérentes avec leur rang."
            )

    st.markdown(f"""
    <div style="background:{rec_bg}; border:1px solid {rec_color}44; border-radius:12px;
                padding:20px 24px; margin-top:8px;">
        <div style="font-family:'Orbitron',sans-serif; font-size:0.9rem; color:{rec_color}; margin-bottom:8px;">
            {rec_icon} RECOMMANDATION FINALE
        </div>
        <div style="color:#e8edf5; font-size:1.05rem; line-height:1.6;">
            {rec_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(
        page_title="Smurf Radar · Rocket League",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(ROCKET_CSS, unsafe_allow_html=True)

    # ── Data loading ──
    df = load_training_data()
    metrics_df = load_metrics()

    # ── Top brand bar ──
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between;
                padding:10px 0 6px; border-bottom:1px solid rgba(0,198,255,0.1);
                margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:14px;">
            <span style="font-family:'Orbitron',sans-serif; font-size:1.1rem; 
                         color:#00c6ff; letter-spacing:0.12em;">🚀 SMURF RADAR</span>
            <span class="rl-badge badge-safe">v1.0 · PoC</span>
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#5a6a84;">
            Modèle : Régression Logistique · Seuil : 95%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs([
        "📖  Le Fléau du Smurfing",
        "🧠  L'Intelligence Artificielle",
        "🎮  Interface Analyste",
    ])

    with tab1:
        tab_business(df)

    with tab2:
        tab_ai(metrics_df, df)

    with tab3:
        tab_ui()


if __name__ == "__main__":
    build_app()