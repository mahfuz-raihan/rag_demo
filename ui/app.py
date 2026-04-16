import os
import chainlit as cl
from typing import List

# --- UI Configuration & Styling ---
@cl.on_chat_start
async def on_chat_start():
    # 1. Welcome Message
    await cl.Message(
        content="👋 **Welcome to your Intelligent Data Assistant!**\n\n"
                "I can analyze your documents (PDF, Word) and calculate insights from your datasets (Excel, CSV).\n\n"
                "To get started, please upload your files."
    ).send()

    # 2. Prompt for File Uploads
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more files to begin analysis.",
            accept=["application/pdf", "text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_size_mb=50,
            max_files=5,
            timeout=180
        ).send()

    # 3. Process Uploaded Files
    msg = cl.Message(content=f"Processing {len(files)} file(s)... ⏳")
    await msg.send()

    file_types = []
    file_paths = []

    for file in files:
        # Extract extension to help our Supervisor Agent later
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in file_types:
            file_types.append(ext)
        file_paths.append(file.path)

    # 4. Initialize Ephemeral Session State
    # This ensures data is isolated per user and clears when they refresh/leave.
    cl.user_session.set("file_paths", file_paths)
    cl.user_session.set("file_types", file_types)
    cl.user_session.set("session_id", cl.user_session.get("id"))
    
    # (In Phase 2, we will trigger the ingestion pipeline here to build the in-memory FAISS/Dataframes)

    msg.content = f"✅ Successfully processed {len(files)} file(s). You can now ask me questions about your data!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    """
    This function handles the user's incoming queries.
    """
    # 1. Retrieve session data
    file_paths = cl.user_session.get("file_paths", [])
    file_types = cl.user_session.get("file_types", [])
    session_id = cl.user_session.get("session_id")

    # 2. Setup initial state for LangGraph
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

    # 3. UI Feedback
    ui_msg = cl.Message(content="🧠 Thinking...")
    await ui_msg.send()

    # --- TODO for Phase 3: Invoke LangGraph here ---
    # async for output in compiled_graph.astream(initial_state):
    #     Update ui_msg based on what node is running (e.g., "📊 Running Data Analyst...")
    # -----------------------------------------------

    # Mock response until the graph is wired up
    mock_response = f"*(Graph not yet wired)*\n\nI received your query: **{message.content}**\nI know you uploaded files of type: `{file_types}`."
    
    ui_msg.content = mock_response
    await ui_msg.update()