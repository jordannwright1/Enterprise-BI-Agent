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
        clean_b64 = b64_string.strip()
        # Handle cases where the string might still have common prefixes
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",")[-1]
            
        img_data = base64.b64decode(clean_b64)
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
            st.cache_data.clear()
            st.cache_resource.clear()
            new_id = str(uuid.uuid4())
            st.session_state.thread_id = new_id
            st.session_state.messages = []
            st.success(f"Brain Wipe Successful. New Thread: {new_id[:8]}")
            st.rerun()
            
        st.markdown("---")
        
        # --- EXAMPLE PROMPTS SECTION ---
        st.subheader("💡 Example Prompts")
        st.info("Click a prompt to copy it, then paste it into the chat.")
        
        example_prompts = [
            "Analyze Bitcoin mining profitability with a dual-axis chart comparing costs vs price points.",
            "Create a radar chart comparing a sustainable coffee brand vs a generic cafe in London.",
            "Analyze market entry into Japan for luxury skincare with a profitability heatmap."
        ]
        
        for ex in example_prompts:
            st.code(ex, language=None)

st.set_page_config(page_title="Navi: Autonomous Agent", page_icon="🌐", layout="wide")

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
Navi is an advanced AI agent that thinks, researches, and executes code in real-time.
""")

st.markdown("---")

render_sidebar()

# Display Messages (Including stored images if any)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            for img_str in msg["images"]:
                display_navi_chart(img_str)

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
        "consecutive_failures": 0,
        "consecutive_research_failures": 0,
        "meditation_triggered": False,
        "history": [HumanMessage(content=prompt)],
        "final_answer": None,
        "last_error": None,
        "generated_tool_code": None,
        "packages": [],
        "image_payload": [],
        "plan": [] 
    }

    with st.chat_message("assistant"):
        final_ans = None
        current_image_payload = []
        is_finished = False
        
        with st.status("Navi is processing...", expanded=True) as status:
            # We use stream mode "updates" to watch the nodes work
            for event in navi_app.stream(initial_input, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    
                    if node_name == "meditator":
                        st.warning("🧘 Navi is meditating on a persistent error...")
                        if "meditation_notes" in node_output:
                            with st.expander("View Root Cause Analysis"):
                                st.write(node_output["meditation_notes"])

                    if "plan" in node_output and node_output["plan"]:
                        plan_text = node_output["plan"][-1]
                        st.write(f"📝 {plan_text}")
                        if "EXIT" in plan_text.upper() or "TERMINATED" in plan_text.upper():
                            is_finished = True
                    
                    if "final_answer" in node_output and node_output["final_answer"]:
                        final_ans = node_output["final_answer"]

                    # CRITICAL FIX: Ensure we accumulate the payload if it's sent
                    if "image_payload" in node_output and node_output["image_payload"]:
                        # Append new images if they aren't already in our local list
                        for img in node_output["image_payload"]:
                            if img not in current_image_payload:
                                current_image_payload.append(img)
            
            # Final Safety Catch: If loop ended but images weren't caught in 'updates',
            # fetch them from the final state of the graph
            final_state = navi_app.get_state(config).values
            if not current_image_payload and final_state.get("image_payload"):
                current_image_payload = final_state.get("image_payload")
            if not final_ans and final_state.get("final_answer"):
                final_ans = final_state.get("final_answer")

            status.update(label="Process Finished", state="complete", expanded=False)

        # --- FINAL RENDERING LOGIC ---
        if final_ans:
            # Detection of placeholders
            placeholders = re.findall(r"\[IMAGE_DATA_HIDDEN_(\d+)\]", final_ans)
            stored_images = []

            # SCENARIO A: INLINE MULTI-IMAGE INJECTION
            if placeholders and current_image_payload:
                split_pattern = r"(?:Figure:\s*|Plot:\s*)?\(?\[IMAGE_DATA_HIDDEN_\d+\]\)?[:\s]*"
                parts = re.split(split_pattern, final_ans)
                
                # Render to UI
                for i, part in enumerate(parts):
                    if part.strip():
                        st.markdown(part.strip())
                    if i < len(current_image_payload):
                        display_navi_chart(current_image_payload[i])
                        stored_images.append(current_image_payload[i])
                
                st.session_state.messages.append({"role": "assistant", "content": final_ans, "images": stored_images})

            # SCENARIO B: SINGLE GENERIC PLACEHOLDER
            elif "[IMAGE_DATA_HIDDEN]" in final_ans and current_image_payload:
                display_text = final_ans.replace("[IMAGE_DATA_HIDDEN]", "").replace("Figure:", "").strip()
                st.markdown(display_text)
                img_to_show = current_image_payload[0]
                display_navi_chart(img_to_show)
                st.session_state.messages.append({"role": "assistant", "content": display_text, "images": [img_to_show]})

            # SCENARIO C: NO PLACEHOLDERS BUT IMAGES EXIST (Append to bottom)
            elif current_image_payload:
                st.markdown(final_ans)
                for img_str in current_image_payload:
                    display_navi_chart(img_str)
                st.session_state.messages.append({"role": "assistant", "content": final_ans, "images": current_image_payload})

            # SCENARIO D: TEXT ONLY
            else:
                st.markdown(final_ans)
                st.session_state.messages.append({"role": "assistant", "content": final_ans})
