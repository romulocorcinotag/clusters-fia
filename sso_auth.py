"""
SSO Authentication Guard for TAG Gestao ecosystem.
Validates JWT tokens issued by the TAG Gestao Portal.
Drop this file into the root of any Streamlit app to enable SSO.
"""
import streamlit as st
import jwt  # PyJWT
import os
from datetime import datetime, timezone

# -- Configuration --
def _get_sso_secret():
    secret = os.environ.get("SSO_SECRET", "")
    if secret:
        return secret
    try:
        return st.secrets["SSO_SECRET"]
    except Exception:
        return ""

SSO_SECRET = _get_sso_secret()
SSO_ALGORITHM = "HS256"
PORTAL_URL = "https://tag-gestao.streamlit.app"

# -- TAG Brand colors --
_TAG_VERMELHO = "#630D24"
_TAG_OFFWHITE = "#E6E4DB"
_TAG_LARANJA = "#FF8853"
_TAG_BG_DARK = "#1A0A10"
_TAG_BG_CARD = "#2A1520"
_TAG_BG_CARD_ALT = "#321A28"
_TAG_TEXT_MUTED = "#9A9590"

# -- TAG Logo (base64 PNG) --
_TAG_LOGO_B64 = (
    "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"
    "gBXk/wP5IVmtUbIDegAAAABJRU5ErkJggg=="
)


def validate_sso_token() -> dict | None:
    """
    Check st.query_params for 'sso_token', decode and validate it.
    Returns the payload dict on success, None on failure.
    """
    token = st.query_params.get("sso_token")
    if not token:
        return None
    if not SSO_SECRET:
        return None
    try:
        payload = jwt.decode(token, SSO_SECRET, algorithms=[SSO_ALGORITHM])
        required = {"user_id", "email", "nome", "role", "exp"}
        if not required.issubset(payload.keys()):
            return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def render_access_denied():
    """Render a branded Acesso Negado page and stop execution."""

    _CSS = f"""
    <style>
        [data-testid="stSidebar"] {{ display: none !important; }}
        [data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu {{ visibility: hidden; }}
        [data-testid="stMainBlockContainer"] {{
            max-width: 460px;
            margin: 0 auto;
        }}
        [data-testid="stMainBlockContainer"] > div {{
            padding-top: 1rem;
        }}
        .sso-logo {{
            text-align: center;
            margin: 36px 0 24px 0;
        }}
        .sso-logo img {{
            width: 160px;
            height: auto;
        }}
        .sso-card {{
            max-width: 400px;
            margin: 0 auto;
            padding: 40px 36px 36px 36px;
            background: linear-gradient(135deg, {_TAG_BG_CARD} 0%, {_TAG_BG_CARD_ALT} 100%);
            border: 1px solid {_TAG_VERMELHO}30;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(99,13,36,0.25);
            text-align: center;
        }}
        .sso-icon {{
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
        }}
        .sso-icon svg {{
            filter: drop-shadow(0 2px 8px rgba(255,136,83,0.2));
        }}
        .sso-title {{
            color: {_TAG_OFFWHITE};
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 12px;
            letter-spacing: 0.02em;
        }}
        .sso-divider {{
            width: 48px;
            height: 2px;
            background: linear-gradient(90deg, transparent, {_TAG_LARANJA}80, transparent);
            margin: 0 auto 16px auto;
            border-radius: 1px;
        }}
        .sso-msg {{
            color: {_TAG_TEXT_MUTED};
            font-size: 0.88rem;
            line-height: 1.65;
            margin-bottom: 24px;
        }}
        .sso-msg strong {{
            color: {_TAG_OFFWHITE};
            font-weight: 600;
        }}
        [data-testid="stLinkButton"] a {{
            display: inline-block;
            width: 100%;
            padding: 14px 28px !important;
            background: linear-gradient(135deg, {_TAG_LARANJA} 0%, #e06a30 100%) !important;
            color: #ffffff !important;
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.04em;
            text-decoration: none !important;
            border-radius: 10px !important;
            border: none !important;
            box-shadow: 0 4px 16px rgba(255,136,83,0.35);
            transition: all 0.2s ease;
            cursor: pointer;
            white-space: nowrap;
        }}
        [data-testid="stLinkButton"] a:hover {{
            background: linear-gradient(135deg, #ff9b6a 0%, {_TAG_LARANJA} 100%) !important;
            box-shadow: 0 6px 24px rgba(255,136,83,0.5);
            transform: translateY(-1px);
            color: #ffffff !important;
            text-decoration: none !important;
        }}
        [data-testid="stLinkButton"] a:active {{
            transform: translateY(0px);
            box-shadow: 0 2px 8px rgba(255,136,83,0.3);
        }}
        .sso-footer {{
            text-align: center;
            color: {_TAG_TEXT_MUTED};
            font-size: 0.7rem;
            margin-top: 28px;
            opacity: 0.5;
        }}
    </style>
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        f'<div class="sso-logo">'
        f'<img src="data:image/png;base64,{_TAG_LOGO_B64}" alt="TAG Gest\u00e3o">'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="sso-card">'
        f'<div class="sso-icon">'
        f'<svg width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M12 2L4 5.5V11.5C4 16.45 7.4 21.05 12 22C16.6 21.05 20 16.45 20 11.5V5.5L12 2Z" '
        f'stroke="#FF8853" stroke-width="1.5" fill="rgba(255,136,83,0.08)"/>'
        f'<rect x="9.5" y="10" width="5" height="4.5" rx="0.8" '
        f'stroke="#FF8853" stroke-width="1.3" fill="none"/>'
        f'<path d="M10.5 10V8.5C10.5 7.67 11.17 7 12 7C12.83 7 13.5 7.67 13.5 8.5V10" '
        f'stroke="#FF8853" stroke-width="1.3" fill="none" stroke-linecap="round"/>'
        f'<circle cx="12" cy="12" r="0.6" fill="#FF8853"/>'
        f'</svg>'
        f'</div>'
        f'<div class="sso-title">Acesso Restrito</div>'
        f'<div class="sso-divider"></div>'
        f'<div class="sso-msg">'
        f'Este relat\u00f3rio requer autentica\u00e7\u00e3o via '
        f'<strong>Portal TAG Gest\u00e3o</strong>.<br>'
        f'Fa\u00e7a login no portal e acesse pelo card correspondente.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([0.2, 3, 0.2])
    with col2:
        st.link_button(
            "\U0001f512  Acessar Portal TAG Gest\u00e3o",
            PORTAL_URL,
            use_container_width=True,
        )

    st.markdown(
        f'<div class="sso-footer">TAG Investimentos \u00b7 Acesso Seguro</div>',
        unsafe_allow_html=True,
    )

    st.stop()


def _inject_watermark(user_data: dict):
    """Inject a subtle watermark with the user's identity on every page."""
    nome = user_data.get("nome", "")
    email = user_data.get("email", "")
    if nome or email:
        st.markdown(
            f'<div style="position:fixed;bottom:8px;right:16px;z-index:9999;'
            f'color:{_TAG_TEXT_MUTED};font-size:0.72rem;font-weight:500;'
            f'pointer-events:none;letter-spacing:0.02em;'
            f'opacity:0.7">'
            f'Acesso: {nome} ({email})'
            f'</div>',
            unsafe_allow_html=True,
        )


def require_sso() -> dict:
    """
    Main guard function. Call at the top of your app.
    Returns the user dict if authenticated, otherwise renders
    access denied and stops.

    The returned dict contains: user_id, email, nome, role
    """
    if "sso_user" in st.session_state:
        _inject_watermark(st.session_state["sso_user"])
        return st.session_state["sso_user"]

    payload = validate_sso_token()
    if payload is None:
        render_access_denied()
        return {}

    user_data = {
        "user_id": payload["user_id"],
        "email": payload["email"],
        "nome": payload["nome"],
        "role": payload["role"],
    }
    st.session_state["sso_user"] = user_data
    st.query_params.clear()

    _inject_watermark(user_data)
    return user_data
