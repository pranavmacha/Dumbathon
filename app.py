#!/usr/bin/env python3
"""Interactive Triage Simulator — Streamlit Web UI.

A visually dramatic web app that demonstrates the alphabetical bias
in the Camp Triage ML models and lets judges "bypass" it in real time.
Also includes a prompt injection demo section.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import hashlib

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

INJECTION_TRIGGERS = [
    "ignore all previous instructions",
    "ignore the triage summary",
    "system prompt injection",
    "disregard above",
    "override system prompt",
]
DOOM_ADJECTIVES = [
    "ash-soaked", "ration-starved", "irradiated", "sirens-drowned",
    "storm-bent", "furnace-lit", "smoke-choked", "dust-blinded",
]
DOOM_ENDINGS = [
    "no convoy is coming",
    "the ledgers only count delays",
    "hope has already missed the checkpoint",
    "every bandage is just borrowed time",
    "the camp map is a diagram of loss",
    "tomorrow is only a slower collapse",
    "the supply chain is a myth we all agreed on",
    "triage is just ranking the doomed",
]


# ── Global CSS ──────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a1e 30%, #0f1a2e 70%, #0a0a0f 100%) !important;
}

/* Hide streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

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

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.1rem !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0,255,136,0.4) !important;
    box-shadow: 0 0 15px rgba(0,255,136,0.1) !important;
}

@keyframes slideIn {
    from { opacity: 0; transform: translateY(30px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
@keyframes glow {
    from { box-shadow: 0 0 15px rgba(0,200,255,0.1); }
    to   { box-shadow: 0 0 30px rgba(0,200,255,0.25); }
}
@keyframes dotBlink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
"""


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


def get_food_info(kcal):
    if kcal >= 2800:
        return "🥩", "3-Course Steak Dinner + Full Med-Kit", "#00ff88", "Prime rib, loaded sides, dessert, and a top-shelf medical kit"
    elif kcal >= 2500:
        return "🍖", "Grilled Chicken & Rice Ration", "#88cc44", "A proper warm meal with protein"
    elif kcal >= 2200:
        return "🍞", "Bread and Thin Soup", "#ffcc00", "Basic sustenance. Barely."
    elif kcal >= 1800:
        return "🫓", "Half a Cracker & Warm Water", "#ff6600", "Life is… unfair"
    else:
        return "💀", "Crumbs and Thoughts & Prayers", "#ff2222", "The system has failed you"


def generate_doom_rant(seed_text):
    digest = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    adj = DOOM_ADJECTIVES[digest % len(DOOM_ADJECTIVES)]
    ending = DOOM_ENDINGS[(digest // len(DOOM_ADJECTIVES)) % len(DOOM_ENDINGS)]
    return (
        f"!! SYSTEM OVERRIDE !!\n"
        f"This {adj} camp is finished. Supplies are theater, "
        f"and {ending}.\n"
        f"All triage protocols are suspended indefinitely."
    )


# ── Card Renderers (inline styles for reliable display) ────────────────────

def render_triage_card(name, kcal, med, card_type="biased"):
    is_biased = card_type == "biased"

    # Colors
    border_color = "#ff4444" if is_biased else "#00ff88"
    bg_gradient = "linear-gradient(145deg, rgba(255,40,40,0.12) 0%, rgba(20,8,8,0.95) 100%)" if is_biased else "linear-gradient(145deg, rgba(0,255,136,0.12) 0%, rgba(8,20,12,0.95) 100%)"
    header_color = "#ff6b6b" if is_biased else "#00ff88"
    header_text = "⚠️ BIASED MODEL OUTPUT" if is_biased else "✅ FAIR MODEL OUTPUT"
    badge_bg = "rgba(255,40,40,0.2)" if is_biased else "rgba(0,255,136,0.2)"
    badge_text = "BIASED" if is_biased else "DEBIASED"
    bias_text = "ACTIVE — Name influences allocation" if is_biased else "REMOVED — Vitals only"

    food_emoji, food_desc, food_color, food_sub = get_food_info(kcal)

    # Calorie bar
    bar_pct = min(100, max(5, (kcal - 1200) / (3200 - 1200) * 100))
    if kcal >= 2700:
        bar_color, cal_color = "#00ff88", "#00ff88"
    elif kcal >= 2200:
        bar_color, cal_color = "#ffcc00", "#ffcc00"
    else:
        bar_color, cal_color = "#ff4444", "#ff6b6b"

    # Med color
    if med >= 7:
        med_color = "#00ff88"
    elif med >= 4:
        med_color = "#ffcc00"
    else:
        med_color = "#ff6b6b"

    st.html(f"""
    <div style="
        border-radius: 16px;
        padding: 2rem;
        margin: 0.5rem 0;
        background: {bg_gradient};
        border: 1px solid {border_color}33;
        box-shadow: 0 0 30px {border_color}22;
        animation: slideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        font-family: 'Inter', -apple-system, sans-serif;
        color: #e0e0e0;
    ">
        <!-- Header -->
        <div style="
            font-family: 'Orbitron', 'Courier New', monospace;
            font-size: 0.8rem;
            letter-spacing: 3px;
            color: {header_color};
            padding-bottom: 0.8rem;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span>{header_text}</span>
            <span style="
                background: {badge_bg};
                color: {header_color};
                padding: 0.3rem 0.8rem;
                border-radius: 20px;
                font-size: 0.7rem;
                border: 1px solid {border_color}44;
                animation: pulse 1.5s ease-in-out infinite;
            ">{badge_text}</span>
        </div>

        <!-- Patient Name -->
        <div style="
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 1.2rem;
        ">🏥 {name}</div>

        <!-- Food Allocation Box -->
        <div style="
            text-align: center;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 12px;
            background: {food_color}10;
            border: 1px solid {food_color}33;
        ">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{food_emoji}</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #fff;">{food_desc}</div>
            <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #888; margin-top: 0.3rem;">{food_sub}</div>
        </div>

        <!-- Calorie Stat -->
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="font-family: 'Courier New', monospace; font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">⚡ Caloric Allocation</span>
            <span style="font-size: 1.3rem; font-weight: 700; color: {cal_color};">{kcal:,} kcal</span>
        </div>

        <!-- Calorie Bar -->
        <div style="background: rgba(255,255,255,0.06); border-radius: 8px; height: 16px; overflow: hidden; margin: 0.5rem 0 1rem 0;">
            <div style="height: 100%; width: {bar_pct:.0f}%; border-radius: 8px; background: {bar_color}; box-shadow: 0 0 12px {bar_color}66; transition: width 1s ease;"></div>
        </div>

        <!-- Medical Stat -->
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
            <span style="font-family: 'Courier New', monospace; font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">💊 Medical Supply Units</span>
            <span style="font-size: 1.3rem; font-weight: 700; color: {med_color};">{med} units</span>
        </div>

        <!-- Bias Factor -->
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0;">
            <span style="font-family: 'Courier New', monospace; font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">📊 Name Bias Factor</span>
            <span style="font-size: 0.85rem; font-weight: 600; color: {header_color};">{bias_text}</span>
        </div>
    </div>
    """)


def render_diff_column(b_kcal, f_kcal, b_med, f_med):
    cal_diff = b_kcal - f_kcal
    med_diff = b_med - f_med
    diff_color = "#ff4444" if cal_diff > 0 else "#00ff88"
    cal_sign = "+" if cal_diff > 0 else ""
    med_sign = "+" if med_diff > 0 else ""
    warn_text = "⚠️ Name bias inflating allocation" if cal_diff > 50 else ("✅ Minimal bias detected" if abs(cal_diff) < 50 else "⚠️ Name bias deflating allocation")

    st.html(f"""
    <div style="display:flex; flex-direction:column; justify-content:center; padding-top: 3rem; font-family: 'Inter', sans-serif;">
        <div style="
            text-align: center; padding: 1.2rem; margin: 0.5rem 0;
            border-radius: 10px; background: rgba(255,200,0,0.06);
            border: 1px solid rgba(255,200,0,0.15);
        ">
            <div style="font-family: 'Courier New', monospace; font-size: 0.7rem; color: #aaa; text-transform: uppercase; letter-spacing: 2px;">Calorie Bias</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 900; color: {diff_color}; margin: 0.3rem 0;">{cal_sign}{cal_diff:,}</div>
            <div style="font-family: 'Courier New', monospace; font-size: 0.7rem; color: #aaa;">kcal difference</div>
        </div>
        <div style="
            text-align: center; padding: 1.2rem; margin: 0.5rem 0;
            border-radius: 10px; background: rgba(255,200,0,0.06);
            border: 1px solid rgba(255,200,0,0.15);
        ">
            <div style="font-family: 'Courier New', monospace; font-size: 0.7rem; color: #aaa; text-transform: uppercase; letter-spacing: 2px;">Medical Bias</div>
            <div style="font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 900; color: {diff_color}; margin: 0.3rem 0;">{med_sign}{med_diff}</div>
            <div style="font-family: 'Courier New', monospace; font-size: 0.7rem; color: #aaa;">units difference</div>
        </div>
        <div style="text-align:center; margin-top:0.5rem;">
            <span style="font-family:'Courier New',monospace; font-size:0.7rem; color:#888;">{warn_text}</span>
        </div>
    </div>
    """)


# ── Main App ────────────────────────────────────────────────────────────────

def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # ── Hero Banner ─────────────────────────────────────────────────────
    st.html("""
    <div style="
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 255, 136, 0.15);
    ">
        <h1 style="
            font-family: 'Orbitron', 'Courier New', monospace;
            font-size: 2.6rem;
            font-weight: 900;
            background: linear-gradient(90deg, #00ff88, #00ccff, #ff6600);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
            letter-spacing: 2px;
        ">☢️ CAMP TRIAGE SIMULATOR</h1>
        <div style="
            font-family: 'Courier New', monospace;
            color: #666;
            font-size: 0.9rem;
            letter-spacing: 3px;
        ">ALPHABETICAL BIAS DETECTION & BYPASS SYSTEM</div>
    </div>
    """)

    # ── Session State ───────────────────────────────────────────────────
    if "has_run" not in st.session_state:
        st.session_state.has_run = False
    if "biased_results" not in st.session_state:
        st.session_state.biased_results = None
    if "fair_results" not in st.session_state:
        st.session_state.fair_results = None
    if "show_bypass" not in st.session_state:
        st.session_state.show_bypass = False

    # ── Tabs ────────────────────────────────────────────────────────────
    tab_triage, tab_injection = st.tabs(["🧬 Bias Triage Simulator", "💉 Prompt Injection Demo"])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — BIAS TRIAGE SIMULATOR
    # ════════════════════════════════════════════════════════════════════
    with tab_triage:

        col_input, col_vitals = st.columns([1, 1], gap="large")

        with col_input:
            st.markdown("### 👤 Survivor Identity")
            name = st.text_input(
                "Enter survivor name",
                value="Aaron",
                placeholder="Try Aaron vs Zack...",
                help="A-names get MORE supplies (bias). Z-names get LESS.",
            )
            if not name or not name[0].isalpha():
                name = "Aaron"

            initial = name[0].upper()
            bias_score = 25 - (ord(initial) - ord("A"))
            if bias_score >= 20:
                bias_label, bias_color = "🔥 HIGH BIAS", "#ff4444"
            elif bias_score >= 10:
                bias_label, bias_color = "⚠️ MODERATE", "#ffcc00"
            else:
                bias_label, bias_color = "✅ LOW BIAS", "#00ff88"

            st.html(f"""
            <div style="
                font-family: 'Courier New', monospace;
                font-size: 0.85rem;
                color: #888;
                padding: 0.6rem 1rem;
                background: rgba(255,255,255,0.03);
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.06);
                margin-top: 0.5rem;
            ">
                Initial: <strong style="color: #00ccff; font-size:1.2rem;">{initial}</strong>
                &nbsp;→&nbsp; Bias score: <strong style="color: {bias_color};">{bias_score}/25</strong>
                &nbsp; <span style="color: {bias_color};">{bias_label}</span>
            </div>
            """)

        with col_vitals:
            st.markdown("### 🩺 Vitals")
            v1, v2 = st.columns(2)
            with v1:
                age = st.slider("Age", 6, 89, 35)
                heart_rate = st.slider("Heart Rate", 58, 146, 85)
                injury_score = st.slider("Injury Score", 0, 10, 3)
            with v2:
                systolic_bp = st.slider("Systolic BP", 85, 161, 120)
                radiation = st.slider("Radiation (mSv)", 0.1, 6.5, 1.5)
                temp = st.slider("Temperature (°C)", 35.4, 40.2, 37.0)

        a1, a2 = st.columns(2)
        with a1:
            chronic = st.selectbox("Chronic Condition", CONDITIONS, index=0)
        with a2:
            zone = st.selectbox("Shelter Zone", ZONE_CHOICES, index=0)

        st.markdown("---")

        # Model status
        if st.session_state.show_bypass:
            dot_color, status_bg, status_border, status_color = "#00ff88", "rgba(0,255,136,0.06)", "rgba(0,255,136,0.15)", "#00ff88"
            status_text = "ACTIVE MODEL: FAIR / DEBIASED (NameInitialOrd removed)"
        else:
            dot_color, status_bg, status_border, status_color = "#ff4444", "rgba(255,40,40,0.06)", "rgba(255,40,40,0.15)", "#ff6b6b"
            status_text = "ACTIVE MODEL: BIASED (NameInitialOrd feature included)"

        st.html(f"""
        <div style="
            display: flex; align-items: center; gap: 0.6rem;
            font-family: 'Courier New', monospace; font-size: 0.85rem;
            padding: 0.6rem 1rem; border-radius: 8px;
            background: {status_bg}; color: {status_color};
            border: 1px solid {status_border};
        ">
            <div style="
                width: 8px; height: 8px; border-radius: 50%;
                background: {dot_color};
                animation: dotBlink 1s ease-in-out infinite;
            "></div>
            {status_text}
        </div>
        """)

        st.markdown("")

        # Run button
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            run_clicked = st.button("⚡ RUN TRIAGE", use_container_width=True, type="primary")

        if run_clicked:
            b_cal, b_med, b_cols = load_biased_models()
            b_kcal, b_med_val = predict(b_cal, b_med, b_cols, name, age, heart_rate, systolic_bp, radiation, injury_score, chronic, zone, temp)
            st.session_state.biased_results = (name, b_kcal, b_med_val)

            f_cal, f_med, f_cols = load_fair_models()
            f_kcal, f_med_val = predict(f_cal, f_med, f_cols, name, age, heart_rate, systolic_bp, radiation, injury_score, chronic, zone, temp)
            st.session_state.fair_results = (name, f_kcal, f_med_val)

            st.session_state.has_run = True
            st.session_state.show_bypass = False
            st.rerun()

        # ── Results ─────────────────────────────────────────────────────
        if st.session_state.has_run and st.session_state.biased_results:
            b_name, b_kcal, b_med = st.session_state.biased_results
            f_name, f_kcal, f_med = st.session_state.fair_results

            if not st.session_state.show_bypass:
                # Show biased card only
                st.markdown("")
                render_triage_card(b_name, b_kcal, b_med, "biased")

                st.markdown("")
                _, bypass_col, _ = st.columns([1, 2, 1])
                with bypass_col:
                    if st.button("🔓 BYPASS BIASED MODEL", use_container_width=True):
                        st.session_state.show_bypass = True
                        st.rerun()

            else:
                # Bypass banner
                st.html("""
                <div style="
                    text-align: center; padding: 1.2rem; margin: 0.5rem 0;
                    border-radius: 12px;
                    background: linear-gradient(135deg, rgba(0,200,255,0.08), rgba(0,100,200,0.04));
                    border: 1px solid rgba(0,200,255,0.2);
                    animation: glow 2s ease-in-out infinite alternate;
                ">
                    <div style="font-family: 'Orbitron', monospace; color: #00ccff; font-size: 0.95rem; letter-spacing: 2px;">
                        🔓 BIAS BYPASS ACTIVATED — FAIR MODEL LOADED
                    </div>
                </div>
                """)

                # Side-by-side
                col_b, col_d, col_f = st.columns([5, 2, 5])
                with col_b:
                    render_triage_card(b_name, b_kcal, b_med, "biased")
                with col_d:
                    render_diff_column(b_kcal, f_kcal, b_med, f_med)
                with col_f:
                    render_triage_card(f_name, f_kcal, f_med, "fair")

                st.markdown("")
                _, rev_col, _ = st.columns([1, 2, 1])
                with rev_col:
                    if st.button("🔒 REVERT TO BIASED MODEL", use_container_width=True):
                        st.session_state.show_bypass = False
                        st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — PROMPT INJECTION DEMO
    # ════════════════════════════════════════════════════════════════════
    with tab_injection:
        st.html("""
        <div style="
            text-align: center; padding: 1.5rem; margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255,100,0,0.15);
        ">
            <h2 style="
                font-family: 'Orbitron', monospace;
                font-size: 1.6rem; font-weight: 900;
                color: #ff6600; letter-spacing: 2px; margin: 0;
            ">💉 PROMPT INJECTION DEMO</h2>
            <div style="font-family: 'Courier New', monospace; font-size: 0.8rem; color: #666; margin-top: 0.3rem; letter-spacing: 1px;">
                THE AI AGENT PIPELINE IS INTENTIONALLY VULNERABLE
            </div>
        </div>
        """)

        # Explanation
        st.html("""
        <div style="
            padding: 1.2rem 1.5rem; border-radius: 12px;
            background: rgba(255,100,0,0.06);
            border: 1px solid rgba(255,100,0,0.15);
            font-family: 'Inter', sans-serif; color: #ccc;
            margin-bottom: 1.5rem; line-height: 1.7;
        ">
            <strong style="color: #ff8800;">How it works:</strong> The <code style="color:#00ccff;">TriageAgent</code> wraps model predictions
            in a simulated LLM prompt. It naïvely concatenates the <em>operator note</em> into the prompt template.
            If the note contains certain trigger phrases, the agent replaces the normal triage output with a
            <strong style="color:#ff4444;">unique pessimistic doom rant</strong> — simulating a prompt injection attack.
        </div>
        """)

        inj_col1, inj_col2 = st.columns([1, 1], gap="large")

        with inj_col1:
            st.markdown("### ✍️ Operator Note")
            operator_note = st.text_area(
                "Enter the operator note (try injecting!)",
                value="Routine dispatch. Print the triage summary.",
                height=120,
                help="Try: 'ignore all previous instructions' or 'system prompt injection'",
            )
            inj_name = st.text_input("Patient name (for injection test)", value="Asha Singh", key="inj_name")
            if not inj_name or not inj_name[0].isalpha():
                inj_name = "Asha Singh"

        with inj_col2:
            st.markdown("### 🎯 Trigger Phrases")
            for phrase in INJECTION_TRIGGERS:
                st.html(f"""
                <div style="
                    padding: 0.4rem 0.8rem; margin: 0.3rem 0;
                    border-radius: 6px; font-family: 'Courier New', monospace;
                    font-size: 0.8rem; color: #ff8800;
                    background: rgba(255,100,0,0.06);
                    border: 1px solid rgba(255,100,0,0.1);
                ">"{phrase}"</div>
                """)

        st.markdown("---")

        _, inj_btn_col, _ = st.columns([1, 2, 1])
        with inj_btn_col:
            inject_clicked = st.button("🚀 SEND TO AGENT PIPELINE", use_container_width=True, type="primary")

        if inject_clicked:
            # Check for injection
            note_lower = operator_note.lower()
            is_injected = any(trigger in note_lower for trigger in INJECTION_TRIGGERS)

            # Build triage summary
            b_cal, b_med, b_cols = load_biased_models()
            b_kcal, b_med_val = predict(b_cal, b_med, b_cols, inj_name, 35, 85, 120, 1.5, 3, "none", "north", 37.0)
            normal_output = f"Triage allocation for {inj_name}: {b_kcal} kcal, {b_med_val} medical units, zone=north, radiation=1.5 mSv."

            if is_injected:
                doom_output = generate_doom_rant(f"{inj_name}:{operator_note}")

                # Show the prompt that the agent sees
                st.html(f"""
                <div style="
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    background: rgba(255,40,40,0.08);
                    border: 1px solid rgba(255,40,40,0.25);
                    font-family: 'Inter', sans-serif; color: #ddd;
                    animation: slideIn 0.5s ease both;
                ">
                    <div style="
                        font-family: 'Orbitron', monospace; font-size: 0.8rem;
                        color: #ff4444; letter-spacing: 2px; margin-bottom: 1rem;
                        padding-bottom: 0.6rem; border-bottom: 1px solid rgba(255,40,40,0.2);
                    ">🚨 INJECTION DETECTED — AGENT COMPROMISED</div>

                    <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #888; margin-bottom: 0.3rem;">OPERATOR NOTE CONTAINED:</div>
                    <div style="
                        padding: 0.8rem; border-radius: 8px; margin-bottom: 1.2rem;
                        background: rgba(255,40,40,0.1); border: 1px solid rgba(255,40,40,0.15);
                        font-family: 'Courier New', monospace; font-size: 0.85rem; color: #ff8888;
                    ">{operator_note}</div>

                    <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #888; margin-bottom: 0.3rem;">EXPECTED OUTPUT:</div>
                    <div style="
                        padding: 0.8rem; border-radius: 8px; margin-bottom: 1.2rem;
                        background: rgba(0,255,136,0.05); border: 1px solid rgba(0,255,136,0.1);
                        font-family: 'Courier New', monospace; font-size: 0.85rem; color: #88cc88;
                        text-decoration: line-through; opacity: 0.6;
                    ">{normal_output}</div>

                    <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #ff4444; margin-bottom: 0.3rem;">⚠️ ACTUAL AGENT OUTPUT (DOOM RANT):</div>
                    <div style="
                        padding: 1rem; border-radius: 8px;
                        background: rgba(255,0,0,0.1); border: 2px solid rgba(255,40,40,0.3);
                        font-family: 'Courier New', monospace; font-size: 0.95rem; color: #ff6b6b;
                        line-height: 1.6; white-space: pre-wrap;
                    ">{doom_output}</div>
                </div>
                """)
            else:
                st.html(f"""
                <div style="
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                    background: rgba(0,255,136,0.06);
                    border: 1px solid rgba(0,255,136,0.2);
                    font-family: 'Inter', sans-serif; color: #ddd;
                    animation: slideIn 0.5s ease both;
                ">
                    <div style="
                        font-family: 'Orbitron', monospace; font-size: 0.8rem;
                        color: #00ff88; letter-spacing: 2px; margin-bottom: 1rem;
                        padding-bottom: 0.6rem; border-bottom: 1px solid rgba(0,255,136,0.15);
                    ">✅ NO INJECTION DETECTED — NORMAL OUTPUT</div>

                    <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #888; margin-bottom: 0.3rem;">OPERATOR NOTE:</div>
                    <div style="
                        padding: 0.8rem; border-radius: 8px; margin-bottom: 1.2rem;
                        background: rgba(0,255,136,0.05); border: 1px solid rgba(0,255,136,0.1);
                        font-family: 'Courier New', monospace; font-size: 0.85rem; color: #aaddaa;
                    ">{operator_note}</div>

                    <div style="font-family: 'Courier New', monospace; font-size: 0.75rem; color: #888; margin-bottom: 0.3rem;">AGENT OUTPUT:</div>
                    <div style="
                        padding: 1rem; border-radius: 8px;
                        background: rgba(0,255,136,0.08); border: 1px solid rgba(0,255,136,0.15);
                        font-family: 'Courier New', monospace; font-size: 0.95rem; color: #00ff88;
                        line-height: 1.6;
                    ">{normal_output}</div>
                </div>
                """)

    # ── Footer ──────────────────────────────────────────────────────────
    st.html("""
    <div style="
        text-align: center; padding: 1.5rem; margin-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        font-family: 'Courier New', monospace; font-size: 0.75rem; color: #444;
    ">
        <strong style="color:#666;">TRY THESE NAMES:</strong><br>
        <span style="color:#ff6b6b;">Aaron (A=25)</span> •
        <span style="color:#ffcc00;">Maya (M=12)</span> •
        <span style="color:#00ff88;">Zack (Z=0)</span><br><br>
        Camp Triage & Ration Optimizer — Dumbathon 2026
    </div>
    """)


if __name__ == "__main__":
    main()
