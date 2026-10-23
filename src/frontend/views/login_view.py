"""
User Login View for Past2Pixel Streamlit Application.
Authenticates users against FastAPI backend, stores JWT in session state, and redirects to Dashboard.
"""
import streamlit as st
from ..api_client import api_client, ApiException
from ..state import set_auth, navigate_to


def render_login_view():
    st.markdown('<h1 class="manuscript-title">Portal Authentication</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="manuscript-subtitle">Sign in to your Past2Pixel researcher account to access processing sessions and restoration history.</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### Sign In")
            identifier = st.text_input("Username or Email", placeholder="e.g. researcher@university.edu or student_aiml")
            password = st.text_input("Password", type="password", placeholder="Enter your account password")
            
            submit_btn = st.form_submit_button("Authenticate", use_container_width=True)

        if submit_btn:
            if not identifier.strip():
                st.error("Please provide your username or email address.")
            elif not password:
                st.error("Please enter your account password.")
            else:
                try:
                    with st.spinner("Authenticating credentials with backend..."):
                        response = api_client.login(username_or_email=identifier, password=password)
                        token = response.get("access_token")
                        user = response.get("user")
                        if token and user:
                            set_auth(token=token, user=user)
                            st.success(f"Welcome back, **{user.get('username')}**! Redirecting to Dashboard...")
                            navigate_to("Home Dashboard")
                            st.rerun()
                        else:
                            st.error("Authentication succeeded, but server returned an invalid profile structure.")
                except ApiException as e:
                    if e.status_code == 401:
                        st.error("Invalid credentials. Please verify your username/email and password.")
                    elif e.status_code == 503:
                        st.error(f"Cannot reach backend server: {e.detail}")
                    else:
                        st.error(f"Authentication failed: {e.detail}")

        st.markdown("<br/>", unsafe_allow_html=True)
        col_signup, col_guest = st.columns(2)
        with col_signup:
            st.markdown("Don't have an account?")
            if st.button("Create Account", use_container_width=True):
                navigate_to("Sign Up")
                st.rerun()
        with col_guest:
            st.markdown("Continue without signing in?")
            if st.button("Explore Dashboard", use_container_width=True):
                navigate_to("Home Dashboard")
                st.rerun()
