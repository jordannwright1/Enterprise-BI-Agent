import streamlit as st
import os
from dotenv import load_dotenv
import uuid
from core.graph import navi_app
from langchain_core.messages import HumanMessage
import re
from core.database import init_db
import sqlite3
import base64
from PIL import Image
import io

# --- 1. CONFIGURATION & ENVIRONMENT 
load_dotenv()
init_db()

def display_navi_chart(b64_string):
    try:
        # Strip potential whitespace or markers
        img_data = base64.b64decode(b64_string.strip())
        img = Image.open(io.BytesIO(img_data))
        st.image(img, caption="Navi's Generated Analysis", use_container_width=True)
    except Exception as e:
        st.error(f"Could not render chart: {e}")

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
            config = {"configurable": {"thread_id": new_id}}
            navi_app.update_state(config, {
                "final_answer": None, 
                "last_error": None, 
                "plan": [], 
                "retry_count": 0,
                "generated_tool_code": None,
                "image_payload": None # Ensure payload is cleared
            })

            print(f"--- 🧠 FULL BRAIN WIPE: Thread {new_id} ---")
            st.rerun()
            
        st.divider()
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
st.markdown("""
Navi is an advanced AI agent that thinks, researches, and executes code in real-time. Navi can identify when it lacks a tool, write its own Python scripts and auto install dependencies, 
self debug errors, research fixes for persistent errors, and create complex data visualizations completely autonomously.
""")

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
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    config["recursion_limit"] = 50 
    
    initial_input = {
        "task": prompt,
        "inventory": ["ddgs_search", "read_file", "write_file"],
        "retry_count": 0,
        "history": [HumanMessage(content=prompt)],
        "final_answer": None,
        "last_error": None,
        "generated_tool_code": None,
        "packages": [],
        "image_payload": None # Initialize image payload
    }

    with st.chat_message("assistant"):
        final_ans = None
        current_image_payload = None
        
        with st.status("Navi is processing...", expanded=True) as status:
            for event in navi_app.stream(initial_input, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    # Progress updates
                    if "plan" in node_output and node_output["plan"]:
                        plan_text = node_output["plan"][-1]
                        st.write(f"📝 {plan_text}")
                    
                    # Capture final answer
                    if "final_answer" in node_output:
                        final_ans = node_output["final_answer"]

                    # Capture the hidden image payload from the state
                    if "image_payload" in node_output and node_output["image_payload"]:
                        current_image_payload = node_output["image_payload"]
            
            status.update(label="Process Finished", state="complete", expanded=False)

        if final_ans:
            # 1. Detect indexed placeholders
            placeholders = re.findall(r"\[IMAGE_DATA_HIDDEN_(\d+)\]", final_ans)
            
            # --- SCENARIO A: INLINE MULTI-IMAGE INJECTION ---
            if placeholders and isinstance(current_image_payload, list):
                # This pattern eats "Figure:", "Plot:", brackets, and stray colons
                split_pattern = r"(?:Figure:\s*|Plot:\s*)?\(?\[IMAGE_DATA_HIDDEN_\d+\]\)?[:\s]*"
                
                parts = re.split(split_pattern, final_ans)
                
                for i, part in enumerate(parts):
                    # Display text segment
                    if part.strip():
                        st.markdown(part.strip())
                    
                    # Inject image at the split point
                    if i < len(current_image_payload):
                        display_navi_chart(current_image_payload[i])
                
                st.session_state.messages.append({"role": "assistant", "content": final_ans})

            # --- SCENARIO B: SINGLE PLACEHOLDER (Backwards Compatibility) ---
            elif "[IMAGE_DATA_HIDDEN]" in final_ans and current_image_payload:
                display_text = final_ans.replace("[IMAGE_DATA_HIDDEN]", "").replace("Figure:", "").strip()
                st.markdown(display_text)
        
                # Handle if it's a list with one item or just a string
                img_to_show = current_image_payload[0] if isinstance(current_image_payload, list) else current_image_payload
                display_navi_chart(img_to_show)
        
                st.session_state.messages.append({"role": "assistant", "content": display_text})

            # --- SCENARIO C: DIRECT BASE64 (Small Result Fallback) ---
            elif re.search(r"(iVBORw0KGgoAAAANSUhEUg[\w\+\/\s\n=]+)", final_ans):
                b64_match = re.search(r"(iVBORw0KGgoAAAANSUhEUg[\w\+\/\s\n=]+)", final_ans)
                chart_data = b64_match.group(1).strip()
                display_text = final_ans.replace(chart_data, "").replace("Figure:", "").strip()
        
                st.markdown(display_text)
                display_navi_chart(chart_data)
        
                st.session_state.messages.append({"role": "assistant", "content": display_text})

            # --- SCENARIO D: TEXT ONLY ---
            else:
                st.markdown(final_ans)
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
                
