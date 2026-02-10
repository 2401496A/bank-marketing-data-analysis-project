import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Application purpose
# ----------------------------
# This Streamlit app deploys a trained Bank Marketing classification model and provides
# a user-friendly interface for entering customer/campaign details and obtaining
# a subscription likelihood prediction + a simple "Call / Do not call" recommendation.

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon=":material/account_balance:",
    layout="wide",
)

# ----------------------------
# Visual theme and background
# ----------------------------
BG_URL = "https://img.freepik.com/premium-photo/black-white-abstract-image-glowing-grid-with-glowing-white-bar-graph-rising-up-from-bottom-right-corner_14117-239359.jpg"
CUSTOM_CSS = f"""
<style>
:root {{
  --bg-overlay: rgba(3, 7, 18, 0.70);
  --card-bg: rgba(255, 255, 255, 0.98);
  --card-bg-alt: rgba(245, 249, 255, 0.98);
  --card-border: rgba(148, 163, 184, 0.30);
  --ink: rgba(15, 23, 42, 0.96);
  --muted: rgba(30, 41, 59, 0.72);
  --brand: rgba(56, 189, 248, 1.0);
  --brand2: rgba(34, 197, 94, 1.0);
  --shadow: 0 18px 46px rgba(0,0,0,0.30);
}}

.stApp {{
  background:
    linear-gradient(var(--bg-overlay), var(--bg-overlay)),
    url("{BG_URL}");
  background-size: cover;
  background-position: center;
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 1.25rem;
  padding-bottom: 2.2rem;
  max-width: 1180px;
}}

section[data-testid="stSidebar"] {{
  background: rgba(2, 6, 23, 0.94);
  border-right: 1px solid rgba(148, 163, 184, 0.22);
}}
section[data-testid="stSidebar"] * {{
  color: rgba(226, 232, 240, 0.95) !important;
}}

h1, h2, h3 {{
  letter-spacing: -0.3px;
  color: rgba(255,255,255,0.96) !important;
}}
p, li, span, div {{
  color: rgba(226,232,240,0.94);
}}

.small-muted {{
  color: rgba(226,232,240,0.84) !important;
  font-size: 0.93rem;
  line-height: 1.35;
}}
.kicker {{
  font-size: 0.95rem;
  font-weight: 800;
  color: rgba(125, 211, 252, 0.98) !important;
}}

div[data-testid="stCaptionContainer"] * {{
  color: rgba(226,232,240,0.86) !important;
}}

  /* Labels always readable */
label, .stMarkdown label, div[data-testid="stWidgetLabel"] label {{
  color: rgba(15, 23, 42, 0.94) !important;
  font-weight: 750 !important;
}}
div[data-testid="stWidgetLabel"] ~ div {{
  color: var(--muted) !important;
}}

  /* Inputs */
div[data-baseweb="select"] > div {{
  border-radius: 12px !important;
}}
div[data-testid="stNumberInput"] input {{
  border-radius: 12px !important;
}}

  /* Tabs: active tab obvious */
div[data-baseweb="tab-list"] {{
  border-bottom: 1px solid rgba(148, 163, 184, 0.30) !important;
  padding-bottom: 2px;
}}
div[data-baseweb="tab"] {{
  font-weight: 900;
  color: rgba(226,232,240,0.85) !important;
  padding: 8px 12px !important;
  border-radius: 999px !important;
}}
div[data-baseweb="tab"][aria-selected="true"] {{
  background: rgba(56, 189, 248, 0.18) !important;
  color: rgba(255,255,255,0.98) !important;
  border-bottom: 4px solid rgba(56, 189, 248, 0.95) !important;
}}

  /* Buttons */
.stButton > button {{
  border-radius: 14px;
  font-weight: 950;
  border: 1px solid rgba(15, 23, 42, 0.18);
  padding: 0.85rem 1.0rem;
}}
.stButton > button[kind="primary"] {{
  background: linear-gradient(90deg, rgba(56,189,248,1), rgba(34,197,94,1)) !important;
  color: white !important;
  border: 0 !important;
  box-shadow: 0 10px 26px rgba(0,0,0,0.22) !important;
}}
.stButton > button[kind="primary"]:hover {{
  filter: brightness(1.05);
  transform: translateY(-1px);
  transition: 120ms ease;
}}

  /* Alerts readable */
div[data-testid="stAlert"] {{
  border-radius: 14px !important;
  border: 1px solid rgba(15, 23, 42, 0.12) !important;
  background: rgba(255, 255, 255, 0.96) !important;
}}
div[data-testid="stAlert"] * {{
  color: rgba(15, 23, 42, 0.96) !important;
  font-weight: 650;
}}

  /* Ensure slider numbers visible on sidebar */
section[data-testid="stSidebar"] div[data-testid="stSlider"] * {{
  color: rgba(226, 232, 240, 0.95) !important;
}}

  /* Make radio look "blue" (avoid default red vibe) */
div[role="radiogroup"] label {{
  border-radius: 12px !important;
}}

  /* Base card style for bordered containers (left card) */
div[data-testid="stVerticalBlockBorderWrapper"]{{
  border: 1px solid rgba(148, 163, 184, 0.30) !important;
  border-radius: 18px !important;
  background: rgba(255, 255, 255, 0.98) !important;
  box-shadow: var(--shadow) !important;
  overflow: hidden;
  position: relative;
}}

div[data-testid="stVerticalBlockBorderWrapper"]::before{{
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  height: 6px;
  width: 100%;
  background: linear-gradient(90deg, rgba(56,189,248,1), rgba(34,197,94,1));
}}

  /* Left card text (dark ink) */
div[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown,
div[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown * {{
  color: rgba(15, 23, 42, 0.96) !important;
}}

  /* Prediction card: dark blue fill like sidebar, bolder text */
div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) {{
  background: rgba(2, 6, 23, 0.94) !important;
  border: 1px solid rgba(148, 163, 184, 0.22) !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) .stMarkdown,
div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) .stMarkdown * {{
  color: rgba(226, 232, 240, 0.96) !important;
  font-weight: 750 !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) h1,
div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) h2,
div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) h3 {{
  color: rgba(255,255,255,0.98) !important;
  font-weight: 900 !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) label,
div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) div[data-testid="stWidgetLabel"] label {{
  color: rgba(226, 232, 240, 0.95) !important;
  font-weight: 850 !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"]:has(#pred-card) div[data-testid="stAlert"] {{
  background: rgba(255, 255, 255, 0.96) !important;
}}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Dropdown options
# ----------------------------
JOB_OPTIONS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]
MARITAL_OPTIONS = ["divorced", "married", "single", "unknown"]
EDU_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
CONTACT_OPTIONS = ["cellular", "telephone", "unknown"]
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["failure", "other", "success", "unknown"]

# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

MODEL_PATH = "bank_logreg_final.joblib"

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(
        "The prediction model could not be loaded. "
        "Please ensure the model file is included in the same folder as the app when deploying.\n\n"
        f"Error details: {e}"
    )
    st.stop()

# ----------------------------
# Feature engineering (must match training)
# ----------------------------
def apply_final_fe(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # campaign bins
    X["campaign_single"] = (X["campaign"] == 1).astype(int)
    X["campaign_few"] = X["campaign"].between(2, 3).astype(int)
    X["campaign_many"] = (X["campaign"] >= 4).astype(int)
    X = X.drop(columns=["campaign"])

    # has_any_loan
    X["has_any_loan"] = ((X["housing"] == "yes") | (X["loan"] == "yes")).astype(int)
    X = X.drop(columns=["housing", "loan"])

    return X

# ----------------------------
# Input validation
# ----------------------------
def validate_inputs(age, balance, day, campaign, pdays, previous):
    errors = []

    if not (18 <= age <= 100):
        errors.append("Age must be between 18 and 100.")
    if not (-10000 <= balance <= 500000):
        errors.append("Account balance looks unrealistic. Use a value between -10,000 and 500,000.")
    if not (1 <= day <= 31):
        errors.append("Contact day must be between 1 and 31.")
    if not (1 <= campaign <= 100):
        errors.append("Number of contacts in this campaign must be between 1 and 100.")
    if not (0 <= pdays <= 10000):
        errors.append("Days since last contact must be between 0 and 10,000.")
    if not (0 <= previous <= 100):
        errors.append("Number of previous contacts must be between 0 and 100.")

    return errors

# ----------------------------
# Build one-row prediction input
# ----------------------------
def build_input_df(
    age: int,
    job: str,
    marital: str,
    education: str,
    balance: int,
    contact: str,
    day: int,
    month: str,
    housing: str,
    loan: str,
    campaign: int,
    previously_contacted: str,
    pdays_if_contacted: int,
    previous: int,
    poutcome: str,
) -> pd.DataFrame:
    if previously_contacted == "No":
        pdays_never = 1
        pdays = 0
        poutcome_final = "unknown"
    else:
        pdays_never = 0
        pdays = int(pdays_if_contacted)
        poutcome_final = poutcome

    return pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "balance": balance,
        "contact": contact,
        "day": day,
        "month": month,
        "housing": housing,
        "loan": loan,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome_final,
        "pdays_never": pdays_never,
    }])

# ----------------------------
# Sidebar controls (confidence cutoff)
# ----------------------------
with st.sidebar:
    st.markdown("## :material/tune: Controls")
    st.write("Use this like a **confidence cutoff** for who to call.")

    threshold = st.slider(
        "Confidence cutoff (when to call)",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01,
        help="Lower = call more people. Higher = call fewer people but with more confidence.",
    )

    st.markdown("---")
    st.markdown("### :material/help: How to use at work")
    st.write("• **Lower cutoff** (e.g., 40%) → broader outreach (more calls)")
    st.write("• **Higher cutoff** (e.g., 70%) → fewer calls, stronger leads")

# ----------------------------
# Header
# ----------------------------
st.markdown('<div class="kicker">Retail Banking • Campaign Targeting Assistant</div>', unsafe_allow_html=True)
st.title(":material/account_balance: Bank Term Deposit Subscription Predictor")
st.caption("Enter customer and campaign details to get a subscription likelihood and a calling recommendation.")

# ----------------------------
# Two-column layout
# ----------------------------
left, right = st.columns([1.08, 1.0], gap="large")

with left:
    with st.container(border=True):
        st.subheader(":material/person: Customer Details")
        st.write(
            '<span class="small-muted">'
            "Inputs are grouped into tabs to keep the form simple and reduce mistakes."
            "</span>",
            unsafe_allow_html=True,
        )

        with st.form("predict_form", clear_on_submit=False):
            tab1, tab2, tab3 = st.tabs(
                [":material/person: Customer", ":material/call: Contact", ":material/leaderboard: Campaign"]
            )

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("Age (years)", min_value=18, max_value=100, value=35, step=1)
                    job = st.selectbox("Occupation group", JOB_OPTIONS, index=JOB_OPTIONS.index("management") if "management" in JOB_OPTIONS else 0)
                    marital = st.selectbox("Marital status", MARITAL_OPTIONS, index=MARITAL_OPTIONS.index("married") if "married" in MARITAL_OPTIONS else 0)
                with c2:
                    education = st.selectbox("Education level", EDU_OPTIONS, index=EDU_OPTIONS.index("secondary") if "secondary" in EDU_OPTIONS else 0)
                    balance = st.number_input("Account balance", min_value=-10000, max_value=500000, value=1500, step=50)
                    housing = st.selectbox("Housing loan?", ["no", "yes"], index=1)
                    loan = st.selectbox("Personal loan?", ["no", "yes"], index=0)

            with tab2:
                c3, c4 = st.columns(2)
                with c3:
                    contact = st.selectbox("Contact channel", CONTACT_OPTIONS, index=CONTACT_OPTIONS.index("cellular") if "cellular" in CONTACT_OPTIONS else 0)
                    month = st.selectbox("Last contact month", MONTH_OPTIONS, index=MONTH_OPTIONS.index("may") if "may" in MONTH_OPTIONS else 0)
                with c4:
                    day = st.slider("Last contact day (1–31)", min_value=1, max_value=31, value=5)

            with tab3:
                c5, c6 = st.columns(2)
                with c5:
                    campaign = st.slider("Contacts in this campaign", min_value=1, max_value=100, value=2)
                    previous = st.slider("Contacts before this campaign", min_value=0, max_value=100, value=0)

                with c6:
                    previously_contacted = st.radio(
                        "Previously contacted before?",
                        ["No", "Yes"],
                        index=0,
                        horizontal=True,
                    )

                    if previously_contacted == "Yes":
                        pdays_if_contacted = st.number_input("Days since last contact", min_value=1, max_value=10000, value=30, step=1)
                        poutcome = st.selectbox("Previous campaign outcome", POUTCOME_OPTIONS, index=POUTCOME_OPTIONS.index("unknown"))
                    else:
                        pdays_if_contacted = 0
                        poutcome = "unknown"

            submitted = st.form_submit_button(
                ":material/analytics: Predict subscription likelihood",
                type="primary",
                use_container_width=True
            )

with right:
    with st.container(border=True):
        # Marker used by CSS to target ONLY this card
        st.markdown('<div id="pred-card" style="display:none;"></div>', unsafe_allow_html=True)

        st.subheader(":material/insights: Prediction Result")

        if not submitted:
            st.error("Fill the tabs on the left, then click Predict.")
        else:
            errors = validate_inputs(
                age=int(age),
                balance=int(balance),
                day=int(day),
                campaign=int(campaign),
                pdays=int(pdays_if_contacted) if previously_contacted == "Yes" else 0,
                previous=int(previous),
            )

            if errors:
                st.error("Please correct the following:")
                for msg in errors:
                    st.write(f"- {msg}")
            else:
                df_input_raw = build_input_df(
                    age=int(age),
                    job=job,
                    marital=marital,
                    education=education,
                    balance=int(balance),
                    contact=contact,
                    day=int(day),
                    month=month,
                    housing=housing,
                    loan=loan,
                    campaign=int(campaign),
                    previously_contacted=previously_contacted,
                    pdays_if_contacted=int(pdays_if_contacted),
                    previous=int(previous),
                    poutcome=poutcome,
                )

                df_input = apply_final_fe(df_input_raw)
                proba = float(model.predict_proba(df_input)[0, 1])
                pred = 1 if proba >= threshold else 0

                proba_pct = proba * 100
                thr_pct = threshold * 100

                st.metric("Likelihood of subscription", f"{proba_pct:.1f}%")

                if pred == 1:
                    st.success("Recommendation: **Target (Call this customer)**")
                    st.write(
                        f"Reason: predicted likelihood (**{proba_pct:.1f}%**) is at least the confidence cutoff (**{thr_pct:.0f}%**)."
                    )
                else:
                    st.warning("Recommendation: **Do not target (Do not call)**")
                    st.write(
                        f"Reason: predicted likelihood (**{proba_pct:.1f}%**) is below the confidence cutoff (**{thr_pct:.0f}%**)."
                    )

                st.caption("Tip: Increase the cutoff to call fewer (higher-confidence) customers, or decrease it to call more customers.")
