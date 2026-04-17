import streamlit as st
import os
from dotenv import load_dotenv
import uuid
from core.graph import navi_app
from langchain_core.messages import HumanMessage
from core.database import init_db
import sqlite3
# --- 1. CONFIGURATION & ENVIRONMENT 

load_dotenv()
init_db()

def inspect_skills():
    conn = sqlite3.connect("tools/navi_skills.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, keyword, created_at FROM skills")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("The database is currently empty.")
    for row in rows:
        print(f"ID: {row[0]} | Skill: {row[1]} | Learned On: {row[2]}")

inspect_skills()

def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        
        # --- RESET MECHANISM ---
        if st.button("🗑️ Reset Agent Memory", use_container_width=True):
        # 1. Clear Streamlit Caching
          st.cache_data.clear()
          st.cache_resource.clear()
    
        # 2. Force a new Thread ID
        new_id = str(uuid.uuid4())
        st.session_state.thread_id = new_id
    
        # 3. Clear UI Messages
        st.session_state.messages = []

        # 4. CRITICAL: Manual State Reset
        # If you want to be 100% sure, manually overwrite the state for the NEW thread
        config = {"configurable": {"thread_id": new_id}}
        navi_app.update_state(config, {
        "final_answer": None, 
        "last_error": None, 
        "plan": [], 
        "retry_count": 0,
        "generated_tool_code": None
    })

        print(f"--- 🧠 FULL BRAIN WIPE: Thread {new_id} ---")
        st.rerun()
            
        st.divider()
        # Visual check: If this ID changes when you click, the button is working.
        current_tid = st.session_state.get('thread_id', 'None')
        st.caption(f"Current Thread: `{current_tid}`")


st.set_page_config(page_title="Navi: Autonomous Agent", page_icon="🌐", layout="wide")

# Ensure API Key is present
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# --- 1. SESSION STATE MANAGEMENT ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "navi_session_001"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. UI LAYOUT ---
st.title("Navi: Self-Learning Multi-Purpose Agent")
st.markdown("---")

with st.sidebar:
    st.header("Navi Status")
    st.info(f"Thread ID: {st.session_state.thread_id}")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = os.urandom(4).hex()
        st.rerun()

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Assign a task to Navi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 3. EXECUTION LOOP ---
    # We removed the Checkpointer config as it's no longer needed for interruptions
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    config["recursion_limit"] = 50 
    
    initial_input = {
        "task": prompt,
        "inventory": ["ddgs_search", "read_file", "write_file"],
        "retry_count": 0,
        "history": [HumanMessage(content=prompt)],
        # Removed 'user_approved' flag
        "final_answer": None,
        "last_error": None,
        "generated_tool_code": None,
        "packages": []
    }

    with st.chat_message("assistant"):
        final_ans = None
        
        with st.status("Navi is processing...", expanded=False) as status:
            # Full autonomous stream - no interrupts
            for event in navi_app.stream(initial_input, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if "plan" in node_output and node_output["plan"]:
                        plan_text = node_output["plan"][-1]
                        st.write(f"📝 {plan_text}")
                    
                    if "final_answer" in node_output and node_output["final_answer"]:
                        final_ans = node_output["final_answer"]
                
                if final_ans:
                    break
            
            status.update(label="Process Finished", state="complete", expanded=False)

        if final_ans:
            st.markdown(final_ans)
            st.session_state.messages.append({"role": "assistant", "content": final_ans})
        else:
            st.error("Navi finished without a final answer. Check your Docker logs or 'core/graph.py' logic.")
