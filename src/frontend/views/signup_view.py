"""
User Registration View for Past2Pixel Streamlit Application.
Validates input fields, invokes registration API, stores JWT, and redirects to Dashboard.
"""
import re
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import set_auth, navigate_to


EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"


def render_signup_view():
    st.markdown('<h1 class="manuscript-title">Researcher Registration</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Create a new Past2Pixel account to isolate and persist manuscript restoration sessions.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form", clear_on_submit=False):
            st.markdown("### Create Account")
            username = st.text_input("Username", placeholder="e.g. anish_aiml (3-50 characters)")
            email = st.text_input("Institutional or Personal Email", placeholder="e.g. researcher@university.edu")
            password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
            confirm_pwd = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")

            submit_btn = st.form_submit_button("Register & Sign In", use_container_width=True)

        if submit_btn:
            u_clean = username.strip()
            e_clean = email.strip()

            if not u_clean or len(u_clean) < 3 or len(u_clean) > 50:
                st.error("Username must be between 3 and 50 characters.")
            elif not re.match(EMAIL_REGEX, e_clean):
                st.error("Please provide a valid email address (e.g. name@domain.com).")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters in length.")
            elif password != confirm_pwd:
                st.error("Passwords do not match. Please re-enter identical passwords.")
            else:
                try:
                    with st.spinner("Registering account and initializing session store..."):
                        response = api_client.register(email=e_clean, username=u_clean, password=password)
                        token = response.get("access_token")
                        user = response.get("user")
                        if token and user:
                            set_auth(token=token, user=user)
                            st.success(f"Account successfully registered for **{user.get('username')}**!")
                            navigate_to("Home Dashboard")
                            st.rerun()
                        else:
                            st.error("Registration completed, but server response was missing tokens.")
                except ApiException as e:
                    if e.status_code == 400:
                        st.error(f"Registration rejected: {e.detail}")
                    elif e.status_code == 503:
                        st.error("Cannot reach backend server. Please verify FastAPI is running.")
                    else:
                        st.error(f"Registration error: {e.detail}")

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("Already have an account?")
        if st.button("Return to Sign In", use_container_width=True):
            navigate_to("Login")
            st.rerun()
