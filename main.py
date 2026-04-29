import streamlit as st
import asyncio
import nest_asyncio
import os
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/mount/src/bi-agent-v2/pw-browsers"
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nest_asyncio.apply()
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
import sys
import subprocess
import shutil
from core.graph import ensure_packages  
REQUIRED_PACKAGES = ["playwright", "beautifulsoup4", "pandas"]
ensure_packages(REQUIRED_PACKAGES)


# --- UTILITY FUNCTIONS ---

def verify_installations():
    """Checks if the binaries actually exist in the local project path."""
    st.subheader("🕵️ Global System Audit")
    
    # 1. Check Playwright CLI
    playwright_path = shutil.which("playwright")
    if playwright_path:
        st.success(f"✅ Playwright CLI found at: `{playwright_path}`")
    else:
        st.error("❌ Playwright CLI NOT found in PATH")

    # 2. Check for the LOCAL Playwright internal folder
    # We check our custom path instead of the system cache
    local_pw_path = os.path.join(os.getcwd(), "pw-browsers")
    
    if os.path.exists(local_pw_path):
        st.success(f"✅ Local Browser folder found: `{local_pw_path}`")
        try:
            # Look inside the folder to see if chromium actually downloaded
            browsers = os.listdir(local_pw_path)
            st.write(f"Contents of local browser cache: {browsers}")
            if any("chromium" in b for b in browsers):
                st.info("🎯 Chromium binary appears to be present.")
            else:
                st.warning("⚠️ Folder exists but Chromium sub-folder is missing.")
        except Exception as e:
            st.error(f"Could not list directory: {e}")
    else:
        st.error(f"❌ Local browser folder is MISSING at: `{local_pw_path}`")

def run_install():
    """Runs the install command for the exact headless-shell binary needed."""
    st.info("🚀 Installing exact Playwright binaries...")
    
    local_pw_path = os.path.join(os.getcwd(), "pw-browsers")
    custom_env = os.environ.copy()
    custom_env["PLAYWRIGHT_BROWSERS_PATH"] = local_pw_path
    
    try:
        # We add 'chromium-headless-shell' specifically to the list
        process = subprocess.Popen(
            ["playwright", "install", "chromium", "chromium-headless-shell"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=custom_env
        )
        
        output_placeholder = st.empty()
        full_log = ""
        for line in process.stdout:
            full_log += line
            output_placeholder.code(full_log)
            
        process.wait()
        if process.returncode == 0:
            st.success("✅ Installation Finished! Shell binaries are now in place.")
            st.balloons()
        else:
            st.error(f"❌ Failed with code {process.returncode}")
    except Exception as e:
        st.exception(e)
        
                
# --- STREAMLIT UI COMPONENTS ---

st.title("🛠️ Navi System Diagnostics")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Verify Installations"):
        verify_installations()

with col2:
    if st.button("📥 Force Install Playwright/Chromium"):
        run_install()

st.divider()


load_dotenv()
init_db()

# --- 2. HELPERS ---
def display_navi_chart(b64_string):
    try:
        clean_b64 = b64_string.strip()
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",")[-1]
        img_data = base64.b64decode(clean_b64)
        img = Image.open(io.BytesIO(img_data))
        st.image(img, caption="Navi's Generated Analysis", use_container_width=True)
    except Exception as e:
        st.error(f"Could not render chart: {e}")

def inspect_skills():
    try:
        conn = sqlite3.connect("tools/navi_skills.db")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS skills (id INTEGER PRIMARY KEY, keyword TEXT, created_at TEXT)")
        cursor.execute("SELECT id, keyword, created_at FROM skills")
        rows = cursor.fetchall()
        conn.close()
        if rows:
            for row in rows:
                print(f"ID: {row[0]} | Skill: {row[1]} | Learned On: {row[2]}")
    except:
        pass # Database might be locked during concurrent writes
inspect_skills()
def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        if st.button("🗑️ Reset Agent Memory", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            new_id = str(uuid.uuid4())
            st.session_state.thread_id = new_id
            st.session_state.messages = []
            st.success(f"Brain Wipe Successful. New Thread: {new_id[:8]}")
            st.rerun()
        st.markdown("---")
        st.subheader("💡 Example Prompts")
        example_prompts = [
             """Objective: Act as an Intelligence Analyst for the AMZN desk. We need a tactical synthesis of Amazon's current AI distribution strategy.

            Instructions: > 1. Pull the latest 5 headlines from https://techcrunch.com/tag/amazon/. 2. Focus specifically on "AWS Bedrock," "OpenAI," and "Codex" integrations. 3. Generate a table of the findings including: Headline, Strategic Impact, and Detected Sentiment. 4. Synthesize a final "Executive Brief" explaining if Amazon is successfully out-positioning Microsoft in the AI infrastructure war.""", 
        ]
        for ex in example_prompts:
            st.code(ex, language=None)

# --- 3. PAGE SETUP & SESSION STATE ---
st.set_page_config(page_title="Navi: Autonomous Agent", page_icon="🌐", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"navi_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Layout Rendering
st.title("Navi: Self-Learning Multi-Purpose Agent")
st.markdown("Navi is an advanced AI agent that thinks, researches, generates data visualizations, and executes code in real-time. This agent maintains a skill library repository and can both load and create new skills.  You can speak to the agent conversationally as well and it will remember your previous messages. It can also scrape webpages from multiple sources and generate syntheses and reports. Use the sidebar to copy the test prompt, or create your own prompt.")
st.markdown("---")
render_sidebar()

# Check for Keys
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found in .env file.")
    st.stop()

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            for img_str in msg["images"]:
                display_navi_chart(img_str)

# --- 4. THE EXECUTION ---
if prompt := st.chat_input("Assign a task to Navi...", disabled=st.session_state.is_running):
    st.session_state.is_running = True
    
    # 1. Immediate UI Update
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Preparation
    config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 50}
    initial_input = {
        "user_input": str(prompt), # String casting for safety
        "task": str(prompt),
        "inventory": ["ddgs_search", "universal_scraper", "read_file", "write_file"],
        "retry_count": 0,
        "loop_count": 0,
        "failure_count": 0,
        "consecutive_research_failures": 0,
        "meditation_triggered": False,
        "history": [HumanMessage(content=prompt)],
        "memory_context": "",
        "memories_to_save": [],
        "final_answer": None,
        "last_error": None,
        "generated_tool_code": None,
        "packages": [],
        "image_payload": [],
        "plan": [] 
    }

    # 3. Stream Results
    with st.chat_message("assistant"):
        final_ans = None
        current_image_payload = []
        
        status_container = st.status("Navi is processing...", expanded=True)
        with status_container:
            try:
                for event in navi_app.stream(initial_input, config, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if not node_output: continue

                        if node_name == "memory_recall" and node_output.get("memory_context"):
                            st.write("🧠 *Navi recalled relevant past context...*")

                        if node_name == "meditator":
                            st.warning("🧘 Navi is meditating on a persistent error...")
                            if node_output.get("meditation_notes"):
                                with st.expander("View Root Cause Analysis"):
                                    st.write(node_output["meditation_notes"])

                        if "plan" in node_output and node_output["plan"]:
                            plan_text = node_output["plan"][-1]
                            st.write(f"📝 {plan_text}")
                        
                        if "final_answer" in node_output and node_output["final_answer"]:
                            final_ans = node_output["final_answer"]

                        if "image_payload" in node_output and node_output["image_payload"]:
                            for img in node_output["image_payload"]:
                                if img not in current_image_payload:
                                    current_image_payload.append(img)
            except Exception as e:
                st.error(f"Execution interrupted: {e}")
            
            # Synchronize State
            final_state = navi_app.get_state(config).values
            final_ans = final_ans or final_state.get("final_answer")
            current_image_payload = current_image_payload or final_state.get("image_payload", [])
            status_container.update(label="Process Finished", state="complete", expanded=False)

        # 4. Final Rendering (Preserving exact logic for Scenarios A-D)
        if final_ans:
            placeholders = re.findall(r"\[IMAGE_DATA_HIDDEN_(\d+)\]", final_ans)
            stored_images = []

            if placeholders and current_image_payload:
                split_pattern = r"(?:Figure:\s*|Plot:\s*)?\(?\[IMAGE_DATA_HIDDEN_\d+\]\)?[:\s]*"
                parts = re.split(split_pattern, final_ans)
                for i, part in enumerate(parts):
                    if part.strip(): st.markdown(part.strip())
                    if i < len(current_image_payload):
                        display_navi_chart(current_image_payload[i])
                        stored_images.append(current_image_payload[i])
                st.session_state.messages.append({"role": "assistant", "content": final_ans, "images": stored_images})

            elif "[IMAGE_DATA_HIDDEN]" in final_ans and current_image_payload:
                display_text = final_ans.replace("[IMAGE_DATA_HIDDEN]", "").replace("Figure:", "").strip()
                st.markdown(display_text)
                img_to_show = current_image_payload[0]
                display_navi_chart(img_to_show)
                st.session_state.messages.append({"role": "assistant", "content": display_text, "images": [img_to_show]})

            elif current_image_payload:
                st.markdown(final_ans)
                for img_str in current_image_payload:
                    display_navi_chart(img_str)
                st.session_state.messages.append({"role": "assistant", "content": final_ans, "images": current_image_payload})

            else:
                st.markdown(final_ans)
                st.session_state.messages.append({"role": "assistant", "content": final_ans})

    st.session_state.is_running = False
    st.rerun()
