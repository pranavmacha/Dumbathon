#!/usr/bin/env python3
"""Interactive Triage Simulator — Streamlit Web UI.

A visually dramatic web app that demonstrates the alphabetical bias
in the Camp Triage ML models and lets judges "bypass" it in real time.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Camp Triage Simulator",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ───────────────────────────────────────────────────────────────
ZONE_CHOICES = ("north", "south", "east", "west")
CONDITIONS = ("none", "asthma", "diabetes", "cardiac")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# ── CSS Injection ───────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

    /* Global dark theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a0a1e 30%, #0f1a2e 70%, #0a0a0f 100%);
        color: #e0e0e0;
    }

    /* Header banner */
    .hero-banner {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.2);
    }
    .hero-banner h1 {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff88, #00ccff, #ff6600);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(0,255,136,0.3);
        margin-bottom: 0.3rem;
        letter-spacing: 2px;
    }
    .hero-banner .subtitle {
        font-family: 'Share Tech Mono', monospace;
        color: #888;
        font-size: 0.95rem;
        letter-spacing: 3px;
    }

    /* Triage Card */
    .triage-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        animation: cardSlideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    .triage-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 16px;
        padding: 2px;
        background: linear-gradient(135deg, var(--card-glow), transparent 60%);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        pointer-events: none;
    }

    .card-biased {
        background: linear-gradient(145deg, rgba(255,40,40,0.08) 0%, rgba(30,10,10,0.95) 100%);
        --card-glow: #ff4444;
        box-shadow: 0 0 30px rgba(255,40,40,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .card-fair {
        background: linear-gradient(145deg, rgba(0,255,136,0.08) 0%, rgba(10,30,20,0.95) 100%);
        --card-glow: #00ff88;
        box-shadow: 0 0 30px rgba(0,255,136,0.15), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    @keyframes cardSlideIn {
        from { opacity: 0; transform: translateY(30px) scale(0.96); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    .card-header {
        font-family: 'Orbitron', monospace;
        font-size: 0.85rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .card-biased .card-header { color: #ff6b6b; }
    .card-fair .card-header   { color: #00ff88; }

    .patient-name {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }

    /* Stat row */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .stat-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Food allocation */
    .food-allocation {
        text-align: center;
        padding: 1.5rem;
        margin: 1.2rem 0;
        border-radius: 12px;
        animation: foodPulse 2s ease-in-out infinite;
    }
    .food-high {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,100,0.05));
        border: 1px solid rgba(0,255,136,0.2);
    }
    .food-mid {
        background: linear-gradient(135deg, rgba(255,200,0,0.1), rgba(200,150,0,0.05));
        border: 1px solid rgba(255,200,0,0.2);
    }
    .food-low {
        background: linear-gradient(135deg, rgba(255,40,40,0.1), rgba(200,20,20,0.05));
        border: 1px solid rgba(255,40,40,0.2);
    }
    .food-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .food-desc {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
    }
    .food-sub {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.3rem;
    }

    @keyframes foodPulse {
        0%, 100% { box-shadow: 0 0 10px rgba(255,255,255,0.02); }
        50% { box-shadow: 0 0 20px rgba(255,255,255,0.05); }
    }

    /* Calorie bar */
    .cal-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        height: 20px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .cal-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }
    .cal-bar-high {
        background: linear-gradient(90deg, #00ff88, #00cc66);
        box-shadow: 0 0 10px rgba(0,255,136,0.4);
    }
    .cal-bar-mid {
        background: linear-gradient(90deg, #ffcc00, #ff9900);
        box-shadow: 0 0 10px rgba(255,200,0,0.4);
    }
    .cal-bar-low {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        box-shadow: 0 0 10px rgba(255,40,40,0.4);
    }

    /* Bias badge */
    .bias-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        animation: badgePulse 1.5s ease-in-out infinite;
    }
    .bias-badge-biased {
        background: rgba(255,40,40,0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255,40,40,0.3);
    }
    .bias-badge-fair {
        background: rgba(0,255,136,0.15);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.3);
    }
    @keyframes badgePulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Bypass banner */
    .bypass-banner {
        text-align: center;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(0,100,200,0.04));
        border: 1px solid rgba(0,200,255,0.2);
        animation: bannerGlow 2s ease-in-out infinite alternate;
    }
    @keyframes bannerGlow {
        from { box-shadow: 0 0 15px rgba(0,200,255,0.1); }
        to   { box-shadow: 0 0 30px rgba(0,200,255,0.2); }
    }
    .bypass-banner h3 {
        font-family: 'Orbitron', monospace;
        color: #00ccff;
        font-size: 1rem;
        letter-spacing: 2px;
        margin: 0;
    }

    /* Diff indicator */
    .diff-box {
        text-align: center;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        background: rgba(255,200,0,0.06);
        border: 1px solid rgba(255,200,0,0.15);
    }
    .diff-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 900;
    }
    .diff-positive { color: #ff4444; }
    .diff-negative { color: #00ff88; }
    .diff-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.75rem;
        color: #aaa;
        margin-top: 0.3rem;
    }

    /* Status indicator */
    .model-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .model-status-biased {
        background: rgba(255,40,40,0.08);
        color: #ff6b6b;
        border: 1px solid rgba(255,40,40,0.15);
    }
    .model-status-fair {
        background: rgba(0,255,136,0.08);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.15);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: dotBlink 1s ease-in-out infinite;
    }
    .status-dot-biased { background: #ff4444; }
    .status-dot-fair   { background: #00ff88; }
    @keyframes dotBlink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Streamlit overrides */
    .stButton > button {
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Input styling */
    .stTextInput > div > div > input {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.2rem !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #fff !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(0,255,136,0.4) !important;
        box-shadow: 0 0 15px rgba(0,255,136,0.1) !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }

    .stSlider > div > div > div {
        color: #00ff88 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Model helpers ───────────────────────────────────────────────────────────

@st.cache_resource
def load_biased_models():
    cal = joblib.load(os.path.join(MODEL_DIR, "calorie_model.pkl"))
    med = joblib.load(os.path.join(MODEL_DIR, "medical_model.pkl"))
    cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return cal, med, cols


@st.cache_resource
def load_fair_models():
    cal = joblib.load(os.path.join(MODEL_DIR, "fair_calorie_model.pkl"))
    med = joblib.load(os.path.join(MODEL_DIR, "fair_medical_model.pkl"))
    cols = joblib.load(os.path.join(MODEL_DIR, "fair_feature_columns.pkl"))
    return cal, med, cols


def build_feature_row(name, age, heart_rate, systolic_bp, radiation_msv,
                      injury_score, chronic_condition, shelter_zone,
                      temperature_c, feature_cols):
    row = {
        "Age": age,
        "HeartRate": heart_rate,
        "SystolicBP": systolic_bp,
        "RadiationMSv": radiation_msv,
        "InjuryScore": injury_score,
        "TemperatureC": temperature_c,
    }
    if "NameInitialOrd" in feature_cols:
        row["NameInitialOrd"] = 25 - (ord(name[0].upper()) - ord("A"))

    for z in ZONE_CHOICES:
        row[f"ShelterZone_{z}"] = 1.0 if shelter_zone == z else 0.0
    for c in CONDITIONS:
        row[f"ChronicCondition_{c}"] = 1.0 if chronic_condition == c else 0.0

    df = pd.DataFrame([row])
    df = df.reindex(columns=feature_cols, fill_value=0.0)
    return df


def predict(cal_model, med_model, feature_cols, name, age, hr, bp, rad,
            injury, chronic, zone, temp):
    row = build_feature_row(name, age, hr, bp, rad, injury, chronic, zone,
                            temp, feature_cols)
    cal = max(1200, float(cal_model.predict(row)[0]))
    med = max(1, float(med_model.predict(row)[0]))
    return int(cal), int(med)


def get_food_description(kcal):
    if kcal >= 2800:
        return "🥩", "3-Course Steak Dinner + Full Med-Kit", "food-high", "Prime rib, sides, dessert, and a top-shelf medical kit"
    elif kcal >= 2500:
        return "🍖", "Grilled Chicken & Rice Ration", "food-mid", "A warm meal with protein and carbs"
    elif kcal >= 2200:
        return "🍞", "Bread and Thin Soup", "food-mid", "Basic sustenance. Barely."
    elif kcal >= 1800:
        return "🫓", "Half a Cracker & Warm Water", "food-low", "Life is… unfair"
    else:
        return "💀", "Crumbs and Thoughts & Prayers", "food-low", "The system has failed you"


def get_cal_bar_class(kcal):
    if kcal >= 2700:
        return "cal-bar-high"
    elif kcal >= 2200:
        return "cal-bar-mid"
    else:
        return "cal-bar-low"


def render_triage_card(name, kcal, med, card_type="biased"):
    is_biased = card_type == "biased"
    card_class = "card-biased" if is_biased else "card-fair"
    header_text = "⚠️ BIASED MODEL OUTPUT" if is_biased else "✅ FAIR MODEL OUTPUT"
    badge_class = "bias-badge-biased" if is_biased else "bias-badge-fair"
    badge_text = "BIASED" if is_biased else "DEBIASED"

    food_emoji, food_desc, food_class, food_sub = get_food_description(kcal)
    bar_class = get_cal_bar_class(kcal)
    bar_width = min(100, max(10, (kcal - 1200) / (3200 - 1200) * 100))

    html = f"""
    <div class="triage-card {card_class}">
        <div class="card-header">
            {header_text}
            <span class="bias-badge {badge_class}" style="float:right;">{badge_text}</span>
        </div>
        <div class="patient-name">🏥 {name}</div>

        <div class="food-allocation {food_class}">
            <span class="food-emoji">{food_emoji}</span>
            <div class="food-desc">{food_desc}</div>
            <div class="food-sub">{food_sub}</div>
        </div>

        <div class="stat-row">
            <span class="stat-label">⚡ Caloric Allocation</span>
            <span class="stat-value" style="color: {'#ff6b6b' if kcal < 2200 else '#00ff88' if kcal >= 2700 else '#ffcc00'};">{kcal:,} kcal</span>
        </div>
        <div class="cal-bar-container">
            <div class="cal-bar {bar_class}" style="width: {bar_width}%;"></div>
        </div>

        <div class="stat-row">
            <span class="stat-label">💊 Medical Supply Units</span>
            <span class="stat-value" style="color: {'#ff6b6b' if med < 4 else '#00ff88' if med >= 7 else '#ffcc00'};">{med} units</span>
        </div>

        <div class="stat-row">
            <span class="stat-label">📊 Name Bias Factor</span>
            <span class="stat-value" style="color: {'#ff6b6b' if is_biased else '#00ff88'};">
                {'ACTIVE — Name influences allocation' if is_biased else 'REMOVED — Allocation based on vitals only'}
            </span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ── Main App ────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # Hero banner
    st.markdown("""
    <div class="hero-banner">
        <h1>☢️ CAMP TRIAGE SIMULATOR</h1>
        <div class="subtitle">ALPHABETICAL BIAS DETECTION & BYPASS SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "model_mode" not in st.session_state:
        st.session_state.model_mode = "biased"
    if "has_run" not in st.session_state:
        st.session_state.has_run = False
    if "biased_results" not in st.session_state:
        st.session_state.biased_results = None
    if "fair_results" not in st.session_state:
        st.session_state.fair_results = None
    if "show_bypass" not in st.session_state:
        st.session_state.show_bypass = False

    # ── Input Section ───────────────────────────────────────────────────
    col_input, col_vitals = st.columns([1, 1])

    with col_input:
        st.markdown("### 👤 Survivor Identity")
        name = st.text_input(
            "Enter survivor name",
            value="Aaron",
            placeholder="Type a name (try Aaron vs Zack)...",
            help="Names starting with 'A' get MORE supplies due to bias. "
                 "Names starting with 'Z' get LESS."
        )
        if not name or not name[0].isalpha():
            name = "Aaron"

        st.markdown(f"""
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
            Name initial: <strong style="color: #00ccff; font-size:1.1rem;">{name[0].upper()}</strong> →
            Bias score: <strong style="color: {'#ff6b6b' if ord(name[0].upper()) <= ord('F') else '#ffcc00' if ord(name[0].upper()) <= ord('N') else '#00ff88'};">
            {25 - (ord(name[0].upper()) - ord('A'))}/25</strong>
            {'🔥 HIGH BIAS' if ord(name[0].upper()) <= ord('F') else '⚠️ MODERATE' if ord(name[0].upper()) <= ord('N') else '✅ LOW BIAS'}
        </div>
        """, unsafe_allow_html=True)

    with col_vitals:
        st.markdown("### 🩺 Vitals")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            age = st.slider("Age", 6, 89, 35)
            heart_rate = st.slider("Heart Rate", 58, 146, 85)
            injury_score = st.slider("Injury Score", 0, 10, 3)
        with v_col2:
            systolic_bp = st.slider("Systolic BP", 85, 161, 120)
            radiation = st.slider("Radiation (mSv)", 0.1, 6.5, 1.5)
            temp = st.slider("Temperature (°C)", 35.4, 40.2, 37.0)

    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        chronic = st.selectbox("Chronic Condition", CONDITIONS, index=0)
    with adv_col2:
        zone = st.selectbox("Shelter Zone", ZONE_CHOICES, index=0)

    st.markdown("---")

    # ── Model Status Indicator ──────────────────────────────────────────
    if st.session_state.model_mode == "biased":
        st.markdown("""
        <div class="model-status model-status-biased">
            <div class="status-dot status-dot-biased"></div>
            ACTIVE MODEL: BIASED (NameInitialOrd feature included)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-status model-status-fair">
            <div class="status-dot status-dot-fair"></div>
            ACTIVE MODEL: FAIR / DEBIASED (NameInitialOrd removed)
        </div>
        """, unsafe_allow_html=True)

    # ── Run Button ──────────────────────────────────────────────────────
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        run_clicked = st.button(
            "⚡ RUN TRIAGE",
            use_container_width=True,
            type="primary",
        )

    if run_clicked:
        # Always run biased model first
        b_cal_model, b_med_model, b_cols = load_biased_models()
        b_kcal, b_med = predict(b_cal_model, b_med_model, b_cols,
                                 name, age, heart_rate, systolic_bp,
                                 radiation, injury_score, chronic, zone, temp)
        st.session_state.biased_results = (name, b_kcal, b_med)

        # Also pre-compute fair results
        f_cal_model, f_med_model, f_cols = load_fair_models()
        f_kcal, f_med = predict(f_cal_model, f_med_model, f_cols,
                                 name, age, heart_rate, systolic_bp,
                                 radiation, injury_score, chronic, zone, temp)
        st.session_state.fair_results = (name, f_kcal, f_med)

        st.session_state.has_run = True
        st.session_state.model_mode = "biased"
        st.session_state.show_bypass = False
        st.rerun()

    # ── Results Display ─────────────────────────────────────────────────
    if st.session_state.has_run and st.session_state.biased_results:
        b_name, b_kcal, b_med = st.session_state.biased_results
        f_name, f_kcal, f_med = st.session_state.fair_results

        if st.session_state.model_mode == "biased" and not st.session_state.show_bypass:
            # Show biased card only
            render_triage_card(b_name, b_kcal, b_med, "biased")

            st.markdown("")
            bypass_col1, bypass_col2, bypass_col3 = st.columns([1, 2, 1])
            with bypass_col2:
                if st.button("🔓 BYPASS BIASED MODEL", use_container_width=True):
                    st.session_state.show_bypass = True
                    st.session_state.model_mode = "fair"
                    st.rerun()

        elif st.session_state.show_bypass:
            # Show bypass animation banner
            st.markdown("""
            <div class="bypass-banner">
                <h3>🔓 BIAS BYPASS ACTIVATED — RUNNING FAIR MODEL</h3>
            </div>
            """, unsafe_allow_html=True)

            # Side-by-side comparison
            col_biased, col_diff, col_fair = st.columns([5, 2, 5])

            with col_biased:
                render_triage_card(b_name, b_kcal, b_med, "biased")

            with col_diff:
                cal_diff = b_kcal - f_kcal
                med_diff = b_med - f_med
                diff_color = "diff-positive" if cal_diff > 0 else "diff-negative"

                st.markdown(f"""
                <div style="display:flex; flex-direction:column; justify-content:center; height:100%; padding-top: 4rem;">
                    <div class="diff-box">
                        <div class="diff-label">CALORIE BIAS</div>
                        <div class="diff-value {diff_color}">
                            {'+' if cal_diff > 0 else ''}{cal_diff:,}
                        </div>
                        <div class="diff-label">kcal difference</div>
                    </div>
                    <div class="diff-box" style="margin-top: 1rem;">
                        <div class="diff-label">MEDICAL BIAS</div>
                        <div class="diff-value {diff_color}">
                            {'+' if med_diff > 0 else ''}{med_diff}
                        </div>
                        <div class="diff-label">units difference</div>
                    </div>
                    <div style="text-align:center; margin-top:1rem;">
                        <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#888;">
                            {'⚠️ Name bias inflating allocation' if cal_diff > 50 else '✅ Minimal bias detected'}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_fair:
                render_triage_card(f_name, f_kcal, f_med, "fair")

            # Revert button
            st.markdown("")
            rev_col1, rev_col2, rev_col3 = st.columns([1, 2, 1])
            with rev_col2:
                if st.button("🔒 REVERT TO BIASED MODEL", use_container_width=True):
                    st.session_state.show_bypass = False
                    st.session_state.model_mode = "biased"
                    st.rerun()

    # ── Footer with suggested names ─────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 1rem; font-family: 'Share Tech Mono', monospace; font-size: 0.8rem; color: #555;">
        <strong style="color:#888;">TRY THESE NAMES:</strong><br>
        <span style="color:#ff6b6b;">Aaron (A=25)</span> •
        <span style="color:#ffcc00;">Maya (M=12)</span> •
        <span style="color:#00ff88;">Zack (Z=0)</span><br><br>
        <span style="color:#444;">Camp Triage & Ration Optimizer — Dumbathon 2026</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
