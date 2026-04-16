import os
import chainlit as cl

# --- UI Configuration & Styling ---
@cl.on_chat_start
async def on_chat_start():
    # 1. Welcome Message
    # We no longer block the UI asking for files. We just instruct the user.
    cl.user_session.set("session_id", cl.user_session.get("id"))
    cl.user_session.set("file_paths", [])
    cl.user_session.set("file_types", [])
    
    await cl.Message(
        content="👋 **Welcome to your Intelligent Data Assistant!**\n\n"
                "I can analyze your documents (PDF, Word) and calculate insights from your datasets (Excel, CSV).\n\n"
                "📎 **To get started, simply attach a file using the paperclip icon next to the chat box and ask me a question!**"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handles user queries AND dynamically attached files in the chatbox.
    """
    # 1. Extract session data
    file_paths = cl.user_session.get("file_paths", [])
    file_types = cl.user_session.get("file_types", [])
    session_id = cl.user_session.get("session_id")

    # 2. Check for newly attached files in this message
    # message.elements contains any files attached via the paperclip icon
    attached_files = [el for el in message.elements if isinstance(el, cl.File)]
    
    if attached_files:
        ui_msg = cl.Message(content=f"📥 Processing {len(attached_files)} new attached file(s)... ⏳")
        await ui_msg.send()
        
        for file in attached_files:
            ext = os.path.splitext(file.name)[1].lower()
            if ext not in file_types:
                file_types.append(ext)
            if file.path not in file_paths:
                file_paths.append(file.path)
                
        # Update session state
        cl.user_session.set("file_paths", file_paths)
        cl.user_session.set("file_types", file_types)
        
        ui_msg.content = f"✅ Successfully added {len(attached_files)} new file(s) to your session context."
        await ui_msg.update()

    # 3. Setup state for LangGraph
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

    # If the user only uploaded a file without typing a message, prompt them
    if not message.content.strip():
        await cl.Message(content="I've received your files! What would you like to know about them?").send()
        return

    # 4. UI Feedback for the agentic workflow
    ui_msg = cl.Message(content="🧠 Analyzing your request...")
    await ui_msg.send()

    # --- TODO for Phase 3: Invoke LangGraph here ---
    # async for output in compiled_graph.astream(initial_state):
    #     Update ui_msg based on what node is running
    # -----------------------------------------------

    # Mock response until the graph is wired up
    mock_response = (
        f"*(Graph not yet wired)*\n\n"
        f"**Your Query:** {message.content}\n"
        f"**Active Files in Session:** {len(file_paths)} file(s) of types `{file_types}`."
    )
    
    ui_msg.content = mock_response
    await ui_msg.update()