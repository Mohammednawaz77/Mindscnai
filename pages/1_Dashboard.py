import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from utils.helpers import Helpers

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

st.title("📊 Dashboard")

user = st.session_state.user
db_manager = st.session_state.db_manager

# User statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sessions = db_manager.get_user_session_count(user['id'])
    st.metric("Total Sessions", total_sessions)

with col2:
    st.metric("Active Models", "5")

with col3:
    st.metric("Channels", "20")

with col4:
    st.metric("Quantum Qubits", "4-8")

st.markdown("---")

# Recent sessions
st.subheader("📁 Recent Sessions")

sessions = db_manager.get_user_sessions(user['id'])

if sessions:
    # Show only last 10 sessions
    for session in sessions[:10]:
        with st.expander(f"{session['filename']} - {Helpers.format_timestamp(session['upload_time'])}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Status:** {session['processing_status']}")
                st.write(f"**Original Channels:** {session['channels_original']}")
                st.write(f"**Selected Channels:** {session['channels_selected']}")
            
            with col2:
                # Get predictions for this session
                predictions = db_manager.get_session_predictions(session['id'])
                if predictions:
                    st.write(f"**Models Run:** {len(predictions)}")
                    best_acc = max([p['accuracy'] for p in predictions if p['accuracy']])
                    st.write(f"**Best Accuracy:** {best_acc:.2%}")
                
                # Get metrics
                metrics = db_manager.get_session_metrics(session['id'])
                if metrics:
                    st.write(f"**Brain State:** {metrics.get('brain_state', 'N/A')}")
else:
    st.info("No sessions found. Upload an EDF file to get started!")

st.markdown("---")

# Activity summary
st.subheader("📈 Recent Activity")

recent_activity = db_manager.get_recent_activity(user['id'], limit=10)

if recent_activity:
    for activity in recent_activity:
        st.text(f"{Helpers.format_timestamp(activity['timestamp'])} - {activity['action']}: {activity['details']}")
else:
    st.info("No recent activity")
