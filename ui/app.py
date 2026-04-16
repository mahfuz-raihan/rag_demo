import os
import sys
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- FIX: Add the root directory to Python's path ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)
# ----------------------------------------------------

import chainlit as cl
from core.graph import compiled_graph
from core.ingestion import process_attached_files

@cl.on_chat_start
async def on_chat_start():
    # Initialize isolated session state
    cl.user_session.set("session_id", cl.user_session.get("id"))
    cl.user_session.set("file_paths", [])
    cl.user_session.set("file_types", [])
    
    await cl.Message(
        content="👋 **Welcome to your Intelligent Data Assistant!**\n\n"
                "I can analyze your documents (PDF, Word) and calculate insights from your datasets (Excel, CSV).\n\n"
                "📎 **To get started, simply attach a file using the paperclip icon next to the chat box and ask me a question!**\n"
                "*(Or just say 'Hi' to test my connection!)*"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handles user queries AND dynamically attached files in the chatbox, 
    routing them through the LangGraph AI workflow.
    """
    file_paths = cl.user_session.get("file_paths", [])
    file_types = cl.user_session.get("file_types", [])
    session_id = cl.user_session.get("session_id")

    # 1. Process any files attached via the paperclip
    attached_files = [el for el in message.elements if isinstance(el, cl.File)]
    
    if attached_files:
        # Show a processing step in the UI
        async with cl.Step(name="System", icon="⚙️") as step:
            step.output = f"Processing {len(attached_files)} attached file(s)..."
            
            # Run ingestion synchronously in the background
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

    # If the user only uploaded a file without typing a message
    if not message.content.strip():
        await cl.Message(content="I've received your files! What would you like to know about them?").send()
        return

    # 2. Setup state for LangGraph
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

    # 3. Create an empty message to stream the final answer into
    ui_msg = cl.Message(content="")
    await ui_msg.send()

    final_answer = "I'm sorry, I couldn't generate an answer."

    # 4. Execute the LangGraph workflow
    # We use a Chainlit Step to visually show the user that the AI is "thinking"
    async with cl.Step(name="Agentic Workflow", icon="🧠") as step:
        step.output = "Routing your query..."
        
        # Invoke the graph
        try:
            # We run the graph synchronously wrapped in make_async to prevent blocking the UI
            result = await cl.make_async(compiled_graph.invoke)(initial_state)
            final_answer = result.get("generation", "No generation found.")
            step.output = f"Workflow complete! Route taken: {result.get('route', 'unknown')}"
        except Exception as e:
            final_answer = f"An error occurred while processing your request: {str(e)}"
            step.output = "Workflow failed."
            step.is_error = True

    # 5. Send the actual LLM response back to the user
    ui_msg.content = final_answer
    await ui_msg.update()