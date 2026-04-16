import os
import sys
import logging
import asyncio

# --- FIX: Silence the HTTPX library used by OpenAI ---
logging.getLogger("httpx").setLevel(logging.WARNING)
# ----------------------------------------------------

# --- FIX: Add the root directory to Python's path ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
# ----------------------------------------------------

import chainlit as cl
from core.graph import compiled_graph
from core.ingestion import process_attached_files

# --- NEW: UI Starters (Suggestions) ---
# This creates the centered layout with vanishing suggestions!
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="📊 Analyze Excel/CSV",
            message="I have attached a dataset. Can you summarize the key metrics and trends?",
        ),
        cl.Starter(
            label="📄 Read Document",
            message="I have attached a PDF. Can you extract the main takeaways?",
        ),
        cl.Starter(
            label="👋 Casual Chat",
            message="Hi! What capabilities do you have?",
        )
    ]
# --------------------------------------

@cl.on_chat_start
async def on_chat_start():
    # Initialize session state quietly (No welcome message sent!)
    cl.user_session.set("session_id", cl.user_session.get("id"))
    cl.user_session.set("file_paths", [])
    cl.user_session.set("file_types", [])

@cl.on_message
async def main(message: cl.Message):
    file_paths = cl.user_session.get("file_paths", [])
    file_types = cl.user_session.get("file_types", [])
    session_id = cl.user_session.get("session_id")

    # 1. Process any attached files
    attached_files = [el for el in message.elements if isinstance(el, cl.File)]
    
    if attached_files:
        async with cl.Step(name="System Loader", icon="⚙️") as step:
            step.output = f"Processing {len(attached_files)} attached file(s)..."
            process_attached_files(attached_files, session_id)
            
            for file in attached_files:
                ext = os.path.splitext(file.name)[1].lower()
                if ext not in file_types:
                    file_types.append(ext)
                if file.path not in file_paths:
                    file_paths.append(file.path)
                    
            cl.user_session.set("file_paths", file_paths)
            cl.user_session.set("file_types", file_types)
            step.output = f"Successfully loaded {len(attached_files)} file(s) into memory."

    if not message.content.strip():
        await cl.Message(content="I've received your files! What would you like to know about them?").send()
        return

    initial_state = {
        "question": message.content,
        "session_id": session_id,
        "uploaded_files": file_paths,
        "file_types": file_types,
        "route": "",
        "retry_count": 0,
        "documents": [],
        "generation": "",
        "reflection": "",
        "dataframe_summaries": ""
    }

    final_answer = "I'm sorry, I couldn't generate an answer."

    # --- Live "Thinking" Animation ---
    async with cl.Step(name="Agentic Workflow", icon="🧠") as step:
        step.output = "Starting workflow..."
        
        try:
            async for output in compiled_graph.astream(initial_state):
                for node_name, node_state in output.items():
                    step.output = f"⚙️ Agent '{node_name.capitalize()}' is processing..."
                    await step.update()
                    
                    if "generation" in node_state:
                        final_answer = node_state["generation"]
            
            step.output = "Workflow complete! Sending response..."
        except Exception as e:
            final_answer = f"An error occurred while processing your request: {str(e)}"
            step.output = "Workflow failed."
            step.is_error = True

    # --- The "Typing" Effect ---
    ui_msg = cl.Message(content="")
    await ui_msg.send()
    
    chunk_size = 3
    for i in range(0, len(final_answer), chunk_size):
        await ui_msg.stream_token(final_answer[i:i+chunk_size])
        await asyncio.sleep(0.01)
        
    await ui_msg.update()