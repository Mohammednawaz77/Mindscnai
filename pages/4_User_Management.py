import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth.authenticator import AuthManager
from utils.helpers import Helpers

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

user = st.session_state.user
db_manager = st.session_state.db_manager

# Check if user is admin
auth_manager = AuthManager(db_manager)

if not auth_manager.is_admin(user):
    st.error("⛔ Access Denied: Admin privileges required")
    st.stop()

st.title("👥 User Management")

# Get all users
users = db_manager.get_all_users()

st.subheader("All Users")

# Users table
user_data = []
for u in users:
    user_data.append({
        'ID': u['id'],
        'Username': u['username'],
        'Full Name': u['full_name'],
        'Email': u['email'],
        'Role': u['role'].title(),
        'Created': Helpers.format_timestamp(u['created_at']),
        'Last Login': Helpers.format_timestamp(u['last_login']) if u['last_login'] else 'Never'
    })

st.table(user_data)

st.markdown("---")

# User statistics
st.subheader("📊 User Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    total_users = len(users)
    st.metric("Total Users", total_users)

with col2:
    admins = sum(1 for u in users if u['role'] == 'admin')
    st.metric("Admins", admins)

with col3:
    researchers = sum(1 for u in users if u['role'] == 'researcher')
    st.metric("Researchers", researchers)

st.markdown("---")

# Add new user (admin only)
st.subheader("➕ Add New User")

with st.form("add_user_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        new_username = st.text_input("Username")
        new_fullname = st.text_input("Full Name")
        new_email = st.text_input("Email")
    
    with col2:
        new_password = st.text_input("Password", type="password")
        new_confirm = st.text_input("Confirm Password", type="password")
        new_role = st.selectbox("Role", ["Admin", "Doctor", "Researcher"])
    
    submitted = st.form_submit_button("Add User", type="primary")
    
    if submitted:
        if new_password != new_confirm:
            st.error("Passwords do not match")
        elif len(new_password) < 6:
            st.error("Password must be at least 6 characters")
        elif not new_username or not new_fullname or not new_email:
            st.error("All fields are required")
        else:
            success, message = auth_manager.register_user(
                new_username, new_password, new_fullname,
                new_email, new_role.lower()
            )
            
            if success:
                st.success(message)
                db_manager.log_activity(
                    user['id'],
                    'user_created',
                    f"Created new user: {new_username} ({new_role})"
                )
                st.rerun()
            else:
                st.error(message)
