import streamlit as st
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth.authenticator import AuthManager
from utils.helpers import Helpers

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

user = st.session_state.user
db_manager = st.session_state.db_manager
auth_manager = AuthManager(db_manager)

st.title("📜 Activity Logs")

# Check permissions
is_admin = auth_manager.is_admin(user)

if is_admin:
    st.subheader("All System Activity (Admin View)")
    
    # Get all activity logs
    limit = st.slider("Number of logs to display", 10, 200, 50)
    
    all_logs = db_manager.get_all_activity_logs(limit=limit)
    
    if all_logs:
        # Convert to DataFrame for better display
        df_data = []
        for log in all_logs:
            df_data.append({
                'Timestamp': Helpers.format_timestamp(log['timestamp']),
                'User': f"{log['full_name']} ({log['username']})",
                'Action': log['action'],
                'Details': log['details']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No activity logs found")

else:
    st.subheader("Your Activity")
    
    # Get user's activity logs
    limit = st.slider("Number of logs to display", 10, 100, 20)
    
    user_logs = db_manager.get_recent_activity(user['id'], limit=limit)
    
    if user_logs:
        # Convert to DataFrame
        df_data = []
        for log in user_logs:
            df_data.append({
                'Timestamp': Helpers.format_timestamp(log['timestamp']),
                'Action': log['action'],
                'Details': log['details']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No activity logs found")

# Activity statistics
st.markdown("---")
st.subheader("📊 Activity Summary")

if is_admin:
    all_logs = db_manager.get_all_activity_logs(limit=1000)
    
    if all_logs:
        # Action type distribution
        action_counts = {}
        for log in all_logs:
            action = log['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Actions", len(all_logs))
        
        with col2:
            st.metric("Unique Actions", len(action_counts))
        
        with col3:
            most_common = max(action_counts.items(), key=lambda x: x[1])[0]
            st.metric("Most Common", most_common)
        
        # Action breakdown
        st.markdown("#### Action Breakdown")
        
        action_df = pd.DataFrame([
            {'Action': k, 'Count': v} 
            for k, v in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.bar_chart(action_df.set_index('Action'))

else:
    user_logs = db_manager.get_recent_activity(user['id'], limit=1000)
    
    if user_logs:
        # Action type distribution
        action_counts = {}
        for log in user_logs:
            action = log['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Your Total Actions", len(user_logs))
        
        with col2:
            st.metric("Action Types", len(action_counts))
        
        # Action breakdown
        st.markdown("#### Your Activity Breakdown")
        
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{action.title()}:** {count}")
