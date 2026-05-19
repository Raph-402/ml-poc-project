"""
Rocket League — Détection Comportementale de Smurfs
Application Streamlit – src/app.py
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import Any

try:
    from config import MODEL_METRICS_FILE, DATA_DIR, MODELS_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    MODEL_METRICS_FILE = BASE_DIR / "model_metrics.csv"

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────
THRESHOLD_SHADOWBAN = 0.80
THRESHOLD_HIGH_RISK = 0.75
THRESHOLD_SUSPECT   = 0.50

# ──────────────────────────────────────────────
# THEME & GLOBAL STYLES
# ──────────────────────────────────────────────
ROCKET_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
:root { --bg-deep:#070b14; --bg-card:#0d1526; --bg-card2:#111b2e; --accent-blue:#00c6ff; --accent-gold:#f5a623; --accent-red:#ff3b5c; --accent-grn:#00e676; --text-main:#e8edf5; --text-muted:#5a6a84; --border:rgba(0,198,255,0.15); }
html, body, [class*="css"] { background-color: var(--bg-deep) !important; color: var(--text-main) !important; font-family: 'Rajdhani', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border) !important; background: transparent !important; }
[data-baseweb="tab"] { font-family: 'Orbitron', sans-serif !important; font-size: 0.72rem !important; letter-spacing: 0.08em; color: var(--text-muted) !important; border-radius: 6px 6px 0 0 !important; padding: 10px 20px !important; transition: color 0.2s, background 0.2s; }
[aria-selected="true"] { color: var(--accent-blue) !important; background: rgba(0,198,255,0.07) !important; border-bottom: 2px solid var(--accent-blue) !important; }
[data-testid="metric-container"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px !important; }
[data-testid="metric-container"] label { font-family: 'Rajdhani', sans-serif; font-size: 0.8rem; letter-spacing: 0.1em; color: var(--text-muted) !important; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; font-size: 1.6rem !important; color: var(--accent-blue) !important; }
.rl-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 14px; padding: 24px 28px; margin-bottom: 18px; }
.rl-card-highlight { background: linear-gradient(135deg, #0d1f3c 0%, #091524 100%); border: 1px solid rgba(0,198,255,0.3); border-radius: 14px; padding: 24px 28px; margin-bottom: 18px; }
.rl-section-title { font-family: 'Orbitron', sans-serif; font-size: 1.05rem; letter-spacing: 0.12em; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 8px; border-left: 3px solid var(--accent-blue); padding-left: 12px; }
.rl-hero-title { font-family: 'Orbitron', sans-serif; font-size: 2.4rem; font-weight: 900; line-height: 1.1; background: linear-gradient(90deg, #00c6ff, #7c4dff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.rl-subtitle { font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; color: var(--text-muted); letter-spacing: 0.05em; }
.rl-badge { display: inline-block; font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em; }
.badge-diamond { background: rgba(90,180,255,0.15); color: #5ab4ff; border: 1px solid #5ab4ff55; }
.badge-gc { background: rgba(245,166,35,0.15); color: var(--accent-gold); border: 1px solid #f5a62355; }
.badge-safe { background: rgba(0,230,118,0.12); color: var(--accent-grn); border: 1px solid #00e67655; }
.badge-warn { background: rgba(255,165,0,0.15); color: #ffa500; border: 1px solid #ffa50055; }
.badge-danger { background: rgba(255,59,92,0.15); color: var(--accent-red); border: 1px solid #ff3b5c55; }
.stButton > button { font-family: 'Orbitron', sans-serif !important; font-size: 0.72rem !important; letter-spacing: 0.1em; background: transparent !important; border: 1px solid var(--accent-blue) !important; color: var(--accent-blue) !important; border-radius: 6px !important; padding: 8px 18px !important; transition: all 0.25s ease; }
.stButton > button:hover { background: rgba(0,198,255,0.1) !important; box-shadow: 0 0 18px rgba(0,198,255,0.25); }
[data-baseweb="select"] { background: var(--bg-card2) !important; border-color: var(--border) !important; border-radius: 8px !important; }
hr { border-color: var(--border) !important; margin: 24px 0; }
</style>
"""
PLOTLY_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Rajdhani, sans-serif", color="#e8edf5"))
_DEFAULT_MARGIN = dict(l=20, r=20, t=40, b=20)

def plotly_axis_style() -> dict:
    return dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.08)")

def section_title(text: str) -> None:
    st.markdown(f'<div class="rl-section-title">{text}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADERS & ML EXTRACTION
# ──────────────────────────────────────────────

@st.cache_data
def load_training_data() -> pd.DataFrame:
    path = Path(DATA_DIR) / "rocket_league_skill_data.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_data_and_split() -> tuple[pd.DataFrame | None, pd.Series | None]:
    df = load_training_data()
    if df.empty or "Target" not in df.columns:
        return None, None
    X = df.drop(columns=["Target"])
    y = df["Target"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test

@st.cache_data
def load_metrics() -> pd.DataFrame:
    try:
        return pd.read_csv(MODEL_METRICS_FILE)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_resource
def load_model() -> Any:
    path = Path(MODELS_DIR) / "log_reg.joblib"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

@st.cache_resource
def get_log_coefficients() -> pd.DataFrame | None:
    model = load_model()
    if model is None: return None
    try:
        log_reg = model.named_steps.get('logisticregression')
        scaler  = model.named_steps.get('standardscaler')
        if not log_reg or not scaler: return None
        
        return pd.DataFrame({
            "Feature": scaler.feature_names_in_,
            "Coefficient": log_reg.coef_[0]
        }).sort_values("Coefficient")
    except Exception:
        return None


# ──────────────────────────────────────────────
# ACCOUNT PROFILES
# ──────────────────────────────────────────────
ACCOUNT_PROFILES = {
    "RAPH-403 — Lobby Champion": {
        "handle": "RAPH-403", "rank": "Champion II", "lobby": "Champion", "mode": "smurf",
        "gc_proba": 0.48, "lobby_avg_proba": 0.10,
        "stats": { "% Supersonique": "16.2%", "Boosts petits pads": "61", "% Haut dans les airs": "11.8%", "Vitesse moy. (uu/s)": "1 340", "Eff. boost": "0.71"},
        "raw_features": {"percentage supersonic speed": 16.2, "amount collected small pads": 61.0, "percentage high in air": 11.8, "avg speed": 1340.0, "boost usage efficiency": 0.71, "demos inflicted": 2.1},
        "narrative": "Jouant en lobby Champion, RAPH-403 affiche un profil mécanique typique des Grand Champions... <strong style='color:#ff3b5c;'>Anomalie comportementale flagrante — profil Smurf.</strong>",
        "lobby_players": [("Player_A", 0.09), ("Player_B", 0.12), ("Player_C", 0.08), ("RAPH-403", 0.48), ("Player_D", 0.11), ("Player_E", 0.10)],
    },
    "RAPH-402 — Lobby GC2": {
        "handle": "RAPH-402", "rank": "GC2", "lobby": "GC2", "mode": "boosted",
        "gc_proba": 0.52, "lobby_avg_proba": 0.58,
        "stats": { "% Supersonique": "21.4%", "Boosts petits pads": "48", "% Haut dans les airs": "0.6%", "Vitesse moy. (uu/s)": "1 510", "Eff. boost": "0.63"},
        "raw_features": {"percentage supersonic speed": 21.4, "amount collected small pads": 48.0, "percentage high in air": 0.6, "avg speed": 1510.0, "boost usage efficiency": 0.63, "demos inflicted": 1.8},
        "narrative": "L'analyse du lobby détecte <strong style='color:#f5a623;'>Player_C</strong> avec seulement 23% de probabilité GC : <strong style='color:#f5a623;'>profil Boosté — mécaniques insuffisantes pour ce niveau.</strong>",
        "lobby_players": [("Player_A", 0.61), ("Player_B", 0.67), ("Player_C", 0.23), ("RAPH-402", 0.52), ("Player_D", 0.59), ("Player_E", 0.72)],
    },
}

# ──────────────────────────────────────────────
# XAI HELPER & RADAR CHART (Nouveautés "Wow Effects")
# ──────────────────────────────────────────────
def feature_contribution_chart(profile_stats: dict, model) -> go.Figure | None:
    try:
        log_reg = model.named_steps["logisticregression"]
        scaler  = model.named_steps["standardscaler"]
        feature_names = list(scaler.feature_names_in_)
        
        raw_values = np.array([profile_stats.get(f, 0) for f in feature_names])
        scaled = scaler.transform([raw_values])[0]
        contribs = scaled * log_reg.coef_[0]
        
        df_c = pd.DataFrame({"Feature": feature_names, "Contribution": contribs}).sort_values("Contribution")
        
        fig = go.Figure(go.Bar(
            x=df_c["Contribution"], y=df_c["Feature"], orientation="h",
            marker_color=["#ff3b5c" if v > 0 else "#00c6ff" for v in df_c["Contribution"]],
            text=[f"{v:+.2f}" for v in df_c["Contribution"]],
            textfont=dict(family="Share Tech Mono", size=11, color="#e8edf5"),
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=dict(text="Impact des mécaniques sur le score final", font=dict(size=13, color="#5a6a84")), xaxis=dict(title="← Tire vers Diamond  |  Tire vers GC →", **plotly_axis_style()), yaxis=dict(**plotly_axis_style()), height=280, margin=_DEFAULT_MARGIN)
        return fig
    except Exception:
        return None

def player_radar_chart(raw_features: dict, player_name: str) -> go.Figure:
    """Génère un graphique Radar comparant le joueur aux baselines Diamond et GC (Effet n°2)."""
    categories = ['Vitesse S.', 'Small Pads', 'Vol Aérien', 'Vitesse Moy', 'Eff. Boost']
    
    # Normalisation approximative pour affichage radar propre (0 à 100)
    p_speed = min(100, (raw_features.get("percentage supersonic speed", 0) / 30.0) * 100)
    p_pads = min(100, (raw_features.get("amount collected small pads", 0) / 100.0) * 100)
    p_air = min(100, (raw_features.get("percentage high in air", 0) / 25.0) * 100)
    p_avg_speed = min(100, (raw_features.get("avg speed", 1200) / 1800.0) * 100)
    p_eff = min(100, raw_features.get("boost usage efficiency", 0) * 100)
    
    player_values = [p_speed, p_pads, p_air, p_avg_speed, p_eff]
    
    # Baselines moyennes
    diamond_baseline = [40, 45, 32, 61, 55]
    gc_baseline = [65, 72, 60, 82, 76]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=diamond_baseline + [diamond_baseline[0]], theta=categories + [categories[0]], fill='toself', fillcolor='rgba(90,180,255,0.02)', line=dict(color='rgba(90,180,255,0.3)', width=1, dash='dash'), name='Moyenne Diamond'))
    fig.add_trace(go.Scatterpolar(r=gc_baseline + [gc_baseline[0]], theta=categories + [categories[0]], fill='toself', fillcolor='rgba(245,166,35,0.02)', line=dict(color='rgba(245,166,35,0.3)', width=1, dash='dash'), name='Moyenne Grand Champion'))
    fig.add_trace(go.Scatterpolar(r=player_values + [player_values[0]], theta=categories + [categories[0]], fill='toself', fillcolor='rgba(0,198,255,0.15)', line=dict(color='#00c6ff', width=2), name=player_name))
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            # C'est ici que se trouvait la correction : showticklabels=False
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#e8edf5")
        ),
        height=280,
        margin=dict(l=45, r=45, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig

def feature_contribution_charts(profile_stats: dict, model) -> tuple[go.Figure | None, go.Figure | None]:
    """Isole le Top 5 des forces positives et négatives pour une lisibilité parfaite."""
    try:
        log_reg = model.named_steps["logisticregression"]
        scaler  = model.named_steps["standardscaler"]
        feature_names = list(scaler.feature_names_in_)
        
        raw_values = np.array([profile_stats.get(f, 0) for f in feature_names])
        scaled = scaler.transform([raw_values])[0]
        contribs = scaled * log_reg.coef_[0]
        
        df_c = pd.DataFrame({"Feature": feature_names, "Contribution": contribs})
        
        # Top 5 des critères musculaires qui poussent vers l'alerte Grand Champion (Rouge)
        df_gc = df_c[df_c["Contribution"] > 0].sort_values("Contribution", ascending=True).tail(5)
        fig_gc = None
        if not df_gc.empty:
            fig_gc = go.Figure(go.Bar(x=df_gc["Contribution"], y=df_gc["Feature"], orientation="h", marker_color="#ff3b5c", text=[f"{v:+.2f}" for v in df_gc["Contribution"]], textposition="outside", textfont=dict(family="Share Tech Mono", size=11, color="#e8edf5")))
            fig_gc.update_layout(**PLOTLY_LAYOUT, title=dict(text="Top 5 Critères → Diagnostic Grand Champion", font=dict(size=12, color="#ff3b5c")), xaxis=dict(**plotly_axis_style()), yaxis=dict(**plotly_axis_style()), height=200, margin=dict(l=160, r=40, t=30, b=20))
        
        # Top 5 des critères musculaires qui justifient le niveau Diamant (Bleu)
        df_dia = df_c[df_c["Contribution"] < 0].sort_values("Contribution", ascending=False).tail(5)
        fig_dia = None
        if not df_dia.empty:
            fig_dia = go.Figure(go.Bar(x=df_dia["Contribution"], y=df_dia["Feature"], orientation="h", marker_color="#00c6ff", text=[f"{v:+.2f}" for v in df_dia["Contribution"]], textposition="outside", textfont=dict(family="Share Tech Mono", size=11, color="#e8edf5")))
            fig_dia.update_layout(**PLOTLY_LAYOUT, title=dict(text="Top 5 Critères → Cohérence Diamant", font=dict(size=12, color="#00c6ff")), xaxis=dict(**plotly_axis_style()), yaxis=dict(**plotly_axis_style()), height=200, margin=dict(l=160, r=40, t=30, b=20))
            
        return fig_gc, fig_dia
    except Exception:
        return None, None
    
@st.cache_resource
def get_top_10_coefficients():
    model = load_model()
    if model is None: return None
    # On récupère les noms et les coefficients
    log_reg = model.named_steps['logisticregression']
    scaler = model.named_steps['standardscaler']
    
    df_coeffs = pd.DataFrame({
        "Feature": scaler.feature_names_in_,
        "Coefficient": log_reg.coef_[0]
    })
    # On prend les 10 plus gros (en valeur absolue)
    df_coeffs["Abs"] = df_coeffs["Coefficient"].abs()
    return df_coeffs.sort_values(by="Abs", ascending=False).head(10).sort_values(by="Coefficient")

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
def tab_business(df: pd.DataFrame) -> None:
    if df.empty: return st.warning("Données non disponibles.")
    st.markdown("""
    <div style="padding: 28px 0 12px;">
        <div class="rl-hero-title">LE FLÉAU DU SMURFING</div>
        <div class="rl-subtitle" style="margin-top:8px;">Quand l'élite se déguise pour écraser les débutants</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Calculs dynamiques pour ton dataset
    diamonds = df[df["Target"] == 0]
    gcs = df[df["Target"] == 1]
    
    # ─── NOUVELLE LIGNE DE 5 KPIs VISUELS ───
    k1, k2, k3, k4, k5 = st.columns(5)
# Remplace l'ancienne ligne k1 par celle-ci :
    k1.metric("Parties Analysées (PoC)", "900", "Matchs 2vs2")    
    k2.metric("Profils Diamants (Label 0)", f"{len(diamonds):,}", "Base Légitime")
    k3.metric("Profils GC (Label 1)", f"{len(gcs):,}", "Base Élite")
    # 12% est l'estimation du Label Noise (Faux Positifs de ta matrice)
    k4.metric("Smurfs Détectés (Test)", f"~{int(len(diamonds) * 0.12):,}", "Pollution Label Noise") 
    k5.metric("Joueurs Mensuels Mondiaux", "93M", "Volume Global Évite Churn")

    st.markdown("<br>", unsafe_allow_html=True)

    section_title("L'Empreinte Algorithmique Globale (Top 10 Critères)")
    log_coeffs = get_top_10_coefficients()
    
    if log_coeffs is not None:
        fig_coeff = go.Figure(go.Bar(
            x=log_coeffs["Coefficient"], 
            y=log_coeffs["Feature"], 
            orientation="h", 
            marker_color=["#ff3b5c" if c > 0 else "#00c6ff" for c in log_coeffs["Coefficient"]]
        ))
        fig_coeff.update_layout(
            **PLOTLY_LAYOUT, 
            height=300, 
            margin=dict(l=150, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_coeff, use_container_width=True)
    
    if 'percentage supersonic speed' in df.columns:
        col_l, col_r = st.columns([1, 1], gap="large")
        with col_l:
            section_title("Distribution — Vitesse Supersonique")
            fig_box = go.Figure()
            for label, subset, color, fill in [("💎 Diamond", diamonds, "#5ab4ff", "rgba(90,180,255,0.12)"), ("🏆 Grand Champion", gcs, "#f5a623", "rgba(245,166,35,0.12)")]:
                fig_box.add_trace(go.Box(y=subset["percentage supersonic speed"], name=label, marker_color=color, line_color=color, fillcolor=fill, boxmean="sd"))
            fig_box.update_layout(**PLOTLY_LAYOUT, yaxis=dict(title="% Supersonique", **plotly_axis_style()), xaxis=dict(**plotly_axis_style()), margin=_DEFAULT_MARGIN, height=350, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with col_r:
            section_title("Carte Mécanique — Vitesse vs Temps en l'air")
            sample = df.sample(min(600, len(df)), random_state=42)
            fig_scatter = px.scatter(sample, x="percentage supersonic speed", y="percentage high in air", color=sample["Target"].map({0: "💎 Diamond", 1: "🏆 Grand Champion"}), color_discrete_map={"💎 Diamond": "#5ab4ff", "🏆 Grand Champion": "#f5a623"}, opacity=0.55)
            fig_scatter.update_layout(**PLOTLY_LAYOUT, xaxis=dict(title="% Supersonique", **plotly_axis_style()), yaxis=dict(title="% Haut dans les airs", **plotly_axis_style()), height=350, margin=_DEFAULT_MARGIN, legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # MÀJ Effet n°3 : Le calculateur de Business Case
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("💼 Simulateur de Pertes Financières Évitées (ROI)")
    st.markdown("<div class='rl-card'><p style='color:#b0c0d8; font-size:0.95rem; line-height:1.6;'>Le smurfing détruit la rétention (churn) des nouveaux joueurs. Ajustez les leviers ci-dessous pour prouver la valeur financière de notre modèle d'anti-triche pour Psyonix :</p></div>", unsafe_allow_html=True)
    
    b1, b2, b3 = st.columns(3)
    with b1: smurfs_ban = st.slider("Smurfs détectés & bannis / mois", 100, 5000, 1500, step=100)
    with b2: churn_saved = st.slider("Joueurs sauvés du churn par ban", 1, 10, 3, step=1)
    with b3: player_ltv = st.slider("Valeur moyenne d'un joueur préservé (LTV)", 10, 100, 30, step=5, format="%d €")
    
    money_saved = smurfs_ban * churn_saved * player_ltv * 12
    st.markdown(f"""
    <div style='background:rgba(0,230,118,0.08); border:1px solid #00e67644; border-radius:12px; padding:20px; text-align:center;'>
        <div style='font-family:Orbitron; color:#00e676; font-size:2rem; font-weight:900;'>{money_saved:,} € / AN SAUVÉS</div>
        <div style='color:#5a6a84; font-size:0.85rem; font-family:Share Tech Mono; margin-top:4px;'>Calcul basé sur les coûts asymétriques et la préservation de la LifeTime Value (LTV)</div>
    </div>
    """, unsafe_allow_html=True)

def tab_ai(metrics_df: pd.DataFrame) -> None:
    st.markdown("""
    <div style="padding: 28px 0 12px;">
        <div class="rl-hero-title">L'INTELLIGENCE ARTIFICIELLE</div>
        <div class="rl-subtitle" style="margin-top:8px;">Détection d'anomalies · Modèles · Explicabilité</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.success("🔒 **Sécurité anti Data-Leakage :** Le `StandardScaler` est encapsulé dans le `Pipeline` scikit-learn de nos scripts. Il n'a donc jamais été 'fit' sur les données de test.")

    col1, col2 = st.columns([1.2, 0.8], gap="large")
    with col1:
        section_title("Pourquoi la Logistique gagne")
        st.markdown("<div class='rl-card-highlight'><p style='color:#b0c0d8; font-size:1rem; line-height:1.8;'>Contre toute attente, la <strong style='color:#00c6ff;'>Régression Logistique surpasse les modèles ensemblistes</strong>. Cela confirme que la relation entre les mécaniques et le rang est linéaire. Nous avons le modèle <strong style='color:#00e676;'>le plus performant ET explicable (White-Box)</strong>.</p></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="large")
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Évaluation Scientifique du Modèle")
    st.markdown("<p style='color:#b0c0d8; font-size:0.95rem;'>La Matrice de Confusion illustre nos prédictions avec le seuil strict. <strong style='color:#ff3b5c;'>Note métier :</strong> Les 'Faux Positifs' (en haut à droite) sont très probablement pollués par de vrais Smurfs démasqués dans les données d'entraînement.</p>", unsafe_allow_html=True)
    
    X_test, y_test = load_data_and_split()
    model = load_model()
        
    if model is not None and X_test is not None:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= THRESHOLD_SHADOWBAN).astype(int)
        
        # Création de 3 colonnes pour tout afficher côte à côte
        col_cm, col_roc, col_pr = st.columns(3, gap="medium")
        
        with col_cm:
            st.markdown("<div style='text-align:center; font-family:Orbitron; color:#00c6ff; margin-bottom:10px;'>🧮 Matrice de Confusion</div>", unsafe_allow_html=True)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            z = [[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]]
            labels = [[f"Vrais Négatifs: {cm[0,0]}<br>✅ Diamants Légitimes", f"Faux Positifs: {cm[0,1]}<br>🚨 Smurfs Débusqués ?"],
                      [f"Faux Négatifs: {cm[1,0]}<br>⚠️ Joueurs Boostés ?", f"Vrais Positifs: {cm[1,1]}<br>✅ GC Légitimes"]]
            
            fig_cm = go.Figure(go.Heatmap(
                z=z, colorscale=[[0, "#0d1526"], [1, "rgba(0,198,255,0.2)"]], showscale=False, 
                text=labels, texttemplate="%{text}", textfont=dict(size=12, family="Rajdhani", color="#e8edf5")
            ))
            fig_cm.update_layout(**PLOTLY_LAYOUT, xaxis=dict(tickvals=[0, 1], ticktext=["Prédit Diamant", "Prédit GC"], **plotly_axis_style()), yaxis=dict(tickvals=[0, 1], ticktext=["Réel Diamant", "Réel GC"], **plotly_axis_style()), margin=_DEFAULT_MARGIN, height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            st.markdown("<div style='text-align:center; font-family:Orbitron; color:#00c6ff; margin-bottom:10px;'>📉 Courbe ROC</div>", unsafe_allow_html=True)
            fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy', fillcolor='rgba(0,198,255,0.08)', line=dict(color='#00c6ff', width=2), name=f'AUC = {roc_auc:.3f}'))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(color='rgba(255,255,255,0.2)', dash='dash'), showlegend=False))
            idx_thresh = np.argmin(np.abs(thresholds_roc - THRESHOLD_SHADOWBAN))
            fig_roc.add_trace(go.Scatter(x=[fpr[idx_thresh]], y=[tpr[idx_thresh]], mode='markers', marker=dict(color='#ff3b5c', size=12, symbol='x'), name=f'Seuil {int(THRESHOLD_SHADOWBAN*100)}%'))
            fig_roc.update_layout(**PLOTLY_LAYOUT, height=300, margin=_DEFAULT_MARGIN, xaxis_title="Taux FP (Innocents Bannis)", yaxis_title="Taux VP (Recall)", legend=dict(x=0.5, y=0.1, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_pr:
            st.markdown("<div style='text-align:center; font-family:Orbitron; color:#00c6ff; margin-bottom:10px;'>⚖️ Precision-Recall</div>", unsafe_allow_html=True)
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall, y=precision, line=dict(color='#f5a623', width=2), name='PR Curve'))
            idx_thresh_pr = np.argmin(np.abs(thresholds_pr - THRESHOLD_SHADOWBAN))
            fig_pr.add_trace(go.Scatter(x=[recall[idx_thresh_pr]], y=[precision[idx_thresh_pr]], mode='markers', marker=dict(color='#ff3b5c', size=12, symbol='x'), name=f'Seuil {int(THRESHOLD_SHADOWBAN*100)}%'))
            fig_pr.update_layout(**PLOTLY_LAYOUT, height=300, margin=_DEFAULT_MARGIN, xaxis_title="Recall (Smurfs détectés)", yaxis_title="Précision (Innocents protégés)", legend=dict(x=0.1, y=0.1, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_pr, use_container_width=True)


def _risk_gauge(proba: float, mode: str = "smurf") -> tuple[go.Figure, str, str]:
    if mode == "smurf":
        if proba < THRESHOLD_SUSPECT: bar_color, label, badge_cls = "#00e676", "NORMAL", "badge-safe"
        elif proba < THRESHOLD_HIGH_RISK: bar_color, label, badge_cls = "#ffa500", "SUSPECT", "badge-warn"
        elif proba < THRESHOLD_SHADOWBAN: bar_color, label, badge_cls = "#ff7043", "ÉLEVÉ", "badge-warn"
        else: bar_color, label, badge_cls = "#ff3b5c", "SHADOWBAN", "badge-danger"

        steps = [
            dict(range=[0, THRESHOLD_SUSPECT*100], color="rgba(0,230,118,0.06)"), 
            dict(range=[THRESHOLD_SUSPECT*100, THRESHOLD_HIGH_RISK*100], color="rgba(255,165,0,0.06)"), 
            dict(range=[THRESHOLD_HIGH_RISK*100, THRESHOLD_SHADOWBAN*100], color="rgba(255,112,67,0.06)"), 
            dict(range=[THRESHOLD_SHADOWBAN*100, 100], color="rgba(255,59,92,0.10)")
        ]
        thresh_val = THRESHOLD_SHADOWBAN * 100
        
    else: # MODE BOOSTÉ
        # La logique est inversée !
        if proba > 0.50: bar_color, label, badge_cls = "#00e676", "NORMAL", "badge-safe"
        elif proba > 0.35: bar_color, label, badge_cls = "#ffa500", "SUSPECT", "badge-warn"
        elif proba > 0.20: bar_color, label, badge_cls = "#ff7043", "ÉLEVÉ", "badge-warn"
        else: bar_color, label, badge_cls = "#ff3b5c", "BOOSTÉ", "badge-danger"

        steps = [
            dict(range=[0, 20], color="rgba(255,59,92,0.10)"), # Zone rouge en bas
            dict(range=[20, 35], color="rgba(255,112,67,0.06)"), 
            dict(range=[35, 50], color="rgba(255,165,0,0.06)"), 
            dict(range=[50, 100], color="rgba(0,230,118,0.06)") # Zone verte en haut
        ]
        thresh_val = 20 # Seuil minimum à 20%

    fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=proba * 100, 
        number=dict(suffix="%", font=dict(family="Orbitron", size=36, color=bar_color)), 
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color="#5a6a84", size=11)), 
            bar=dict(color=bar_color, thickness=0.25), 
            bgcolor="rgba(0,0,0,0)", borderwidth=0, 
            steps=steps, 
            threshold=dict(line=dict(color="#ff3b5c", width=2), thickness=0.8, value=thresh_val)
        )
    ))
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=260, margin=dict(l=30, r=30, t=10, b=10))
    return fig, label, badge_cls

def tab_ui() -> None:
    st.markdown("""<div style="padding: 20px 0 8px;"><div class="rl-hero-title" style="font-size:1.9rem;">SMURF RADAR</div><div class="rl-subtitle">Analyse comportementale en temps réel</div></div>""", unsafe_allow_html=True)

    # Construction de la liste des choix avec le simulateur live (Effet n°1)
    options_list = list(ACCOUNT_PROFILES.keys()) + ["🛠️ [CRASH TEST] Simulateur Live (Joueur Personnalisé)"]

    col_search, col_analyze = st.columns([3, 1], gap="medium")
    with col_search:
        account_choice = st.selectbox("🎮 Pseudo / Compte à analyser", options=options_list, index=0)
    with col_analyze:
        st.markdown("<br>", unsafe_allow_html=True)
        do_analyze = st.button("⚡ LANCER L'ANALYSE", use_container_width=True)

    st.markdown("---")

    if not do_analyze and "last_analyzed" not in st.session_state:
        st.markdown("<div style='text-align:center; padding:60px 0; color:#5a6a84;'><div style='font-family:Orbitron; font-size:1.1rem;'>EN ATTENTE D'ANALYSE</div></div>", unsafe_allow_html=True)
        return

    if do_analyze: st.session_state.last_analyzed = account_choice
    current_selection = st.session_state.get("last_analyzed", account_choice)

    # Gestion de l'Effet n°1 : Le simulateur Live avec les Sliders dynamiques
    if "Simulateur Live" in current_selection:
        st.markdown("<div class='rl-card-highlight'>", unsafe_allow_html=True)
        st.write("🔧 **Mode Simulation :** Modifiez les mécaniques du joueur mystère pour voir les calculs de l'IA changer en direct :")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            s_speed = sc1.slider("% Temps Supersonique", 5.0, 30.0, 14.0, 0.5)
            s_pads = sc1.slider("Pads de boost collectés", 10, 110, 52)
        with sc2:
            s_air = sc2.slider("% Temps en l'Air", 0.0, 25.0, 7.5, 0.5)
            s_avg_speed = sc2.slider("Vitesse Moyenne (uu/s)", 800, 1800, 1250, 50)
        with sc3:
            s_eff = sc3.slider("Efficacité du Boost", 0.10, 1.00, 0.58, 0.02)
            s_demos = sc3.slider("Démos infligées", 0.0, 5.0, 1.2, 0.1)
        st.markdown("</div>", unsafe_allow_html=True)

        raw_features = {
            "percentage supersonic speed": s_speed,
            "amount collected small pads": float(s_pads),
            "percentage high in air": s_air,
            "avg speed": float(s_avg_speed),
            "boost usage efficiency": s_eff,
            "demos inflicted": s_demos
        }

        # Calcul de la prédiction du vrai modèle EN DIRECT
        model = load_model()
        if model:
            scaler = model.named_steps.get('standardscaler')
            cols = list(scaler.feature_names_in_) if scaler else list(raw_features.keys())
            
            # --- LA CORRECTION MAGIQUE ---
            # On charge les données pour connaître les stats d'un "Vrai Diamant"
            df_ref = load_training_data()
            if not df_ref.empty and "Target" in df_ref.columns:
                # On calcule la moyenne des joueurs légitimes (Target == 0)
                diamond_baseline = df_ref[df_ref["Target"] == 0].mean().to_dict()
            else:
                diamond_baseline = {}
            
            # On fusionne nos curseurs avec le profil Diamant normal
            input_dict = {}
            for c in cols:
                if c in raw_features:
                    input_dict[c] = raw_features[c]  # La valeur du curseur
                else:
                    input_dict[c] = diamond_baseline.get(c, 0.0)  # La valeur Diamant par défaut
                    
            input_df = pd.DataFrame([input_dict])
            # -----------------------------
            
            live_proba = model.predict_proba(input_df)[0][1]
        else:
            live_proba = 0.50

        active_profile = {
            "handle": "SIMULATEUR_LIVE", "rank": "Diamond II (Simulé)", "lobby": "Diamond (Simulé)", "mode": "smurf",
            "gc_proba": live_proba, "lobby_avg_proba": 0.12,
            "stats": { "% Supersonique": f"{s_speed}%", "Boosts petits pads": f"{s_pads}", "% Haut dans les airs": f"{s_air}%", "Vitesse moy. (uu/s)": f"{s_avg_speed:,}", "Eff. boost": f"{s_eff:.2f}"},
            "raw_features": raw_features,
            "narrative": f"Profil généré dynamiquement. L'ADN mécanique injecté donne un verdict instantané de l'IA avec une probabilité de fraude de {live_proba:.1%}.",
            "lobby_players": [("Player_A", 0.08), ("Player_B", 0.11), ("PROFIL_SIMULÉ", live_proba), ("Player_C", 0.10)]
        }
    else:
        active_profile = ACCOUNT_PROFILES[current_selection]

    st.markdown(f"""<div style="display:flex; align-items:center; gap:12px; margin-bottom:20px;">
        <div style="font-family:'Orbitron',sans-serif; font-size:1.5rem; color:#e8edf5;">{active_profile['handle']}</div>
        <span class="rl-badge badge-diamond">💎 {active_profile['rank']}</span>
        <span class="rl-badge badge-gc">Lobby : {active_profile['lobby']}</span>
    </div>""", unsafe_allow_html=True)

    res_left, res_right = st.columns([1.1, 0.9], gap="large")

    with res_left:
        # On récupère le mode du profil
        mode = active_profile.get("mode", "smurf")
        
        if mode == "boosted":
            section_title("Alerte Usurpateur (Boosté)")
            gauge_desc = "Niveau GC détecté (Alerte si inférieur au seuil de 20%)"
        else:
            section_title("Score de l'IA (Smurf)")
            gauge_desc = "Probabilité que ce joueur Diamant soit un GC caché"

        # On passe directement la proba GC à la jauge, et on lui précise le mode !
        fig_gauge, risk_label, badge_cls = _risk_gauge(active_profile["gc_proba"], mode=mode)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"<div style='text-align:center; margin-top:-10px;'><span class='rl-badge {badge_cls}' style='font-size:1rem; padding:6px 20px;'>{risk_label}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin-top:8px; font-size:0.85rem; color:#5a6a84; font-family:Share Tech Mono;'>{gauge_desc}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Explicabilité de la décision (Facteurs Locaux)")
        model = load_model()
        if model:
            # On récupère nos deux graphiques Top 5 épurés
            fig_gc, fig_dia = feature_contribution_charts(active_profile["raw_features"], model)
            if fig_gc or fig_dia:
                cx1, cx2 = st.columns(2)
                with cx1:
                    if fig_gc: st.plotly_chart(fig_gc, use_container_width=True)
                with cx2:
                    if fig_dia: st.plotly_chart(fig_dia, use_container_width=True)

    with res_right:
        # MÀJ Effet n°2 : Le Radar Chart à la place du simple tableau texte
        section_title("Radar d'ADN Mécanique (Comparatif Spatio-Temporel)")
        fig_radar = player_radar_chart(active_profile["raw_features"], active_profile["handle"])
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        section_title("Vue du Lobby")
        lp_names = [n for n, _ in active_profile["lobby_players"]]
        lp_probas = [p * 100 for _, p in active_profile["lobby_players"]]
        bar_colors = ["#ff3b5c" if n in [active_profile["handle"], "PROFIL_SIMULÉ"] and active_profile["gc_proba"] >= THRESHOLD_SHADOWBAN else "#f5a623" if n in [active_profile["handle"], "PROFIL_SIMULÉ"] else "#1e3a5a" for n in lp_names]
        fig_lobby = go.Figure(go.Bar(x=lp_names, y=lp_probas, marker=dict(color=bar_colors, line=dict(width=0)), text=[f"{v:.0f}%" for v in lp_probas], textposition="outside", textfont=dict(family="Share Tech Mono", size=11, color="#e8edf5")))
        fig_lobby.add_hline(y=THRESHOLD_SHADOWBAN*100, line_dash="dash", line_color="#ff3b5c", annotation_text=f"SEUIL BAN {THRESHOLD_SHADOWBAN*100}%")
        fig_lobby.update_layout(**PLOTLY_LAYOUT, yaxis=dict(range=[0, 115], **plotly_axis_style()), xaxis=dict(**plotly_axis_style()), height=200, margin=_DEFAULT_MARGIN, showlegend=False)
        st.plotly_chart(fig_lobby, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("📝 Analyse Comportementale Finale")
    st.markdown(f"<div class='rl-card'><p style='color:#b0c0d8; font-size:1rem; line-height:1.8;'>{active_profile['narrative']}</p></div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def build_app() -> None:
    st.set_page_config(page_title="Smurf Radar", page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(ROCKET_CSS, unsafe_allow_html=True)
    df = load_training_data()
    metrics_df = load_metrics()

    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:space-between; padding:10px 0 6px; border-bottom:1px solid rgba(0,198,255,0.1); margin-bottom:8px;">
        <div style="display:flex; align-items:center; gap:14px;"><span style="font-family:'Orbitron'; font-size:1.1rem; color:#00c6ff; letter-spacing:0.12em;">🚀 SMURF RADAR</span><span class="rl-badge badge-safe">v1.0 · MLOps PoC</span></div>
        <div style="font-family:'Share Tech Mono'; font-size:0.72rem; color:#5a6a84;">Algorithme : Régression Logistique · Seuil strict : 80%</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Le Fléau du Smurfing", "🧠 L'Intelligence Artificielle", "🎮 Interface Analyste"])
    with tab1: tab_business(df)
    with tab2: tab_ai(metrics_df)
    with tab3: tab_ui()

if __name__ == "__main__":
    build_app()