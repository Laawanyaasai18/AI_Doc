import os
import gradio as gr
import base64
import asyncio
from app.core.document_processor import chunk_documents, get_embeddings
from app.core.retrieval import load_vectorstore
from app.core.groq_client import create_medical_qa_chain, answer_medical_question
from langchain_community.document_loaders import PyPDFLoader

# Global state
qa_chain = None
fallback_chain = None

def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"

logo_base64 = encode_image_to_base64("data/logo.png")

def upload_pdfs_inline(files):
    global qa_chain
    if not files:
        return "", [{"role": "assistant", "content": "❌ No file selected."}]
    try:
        all_chunks = []
        uploaded_files = []
        for path in files:
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
            uploaded_files.append(os.path.basename(path))

        embeddings = get_embeddings()
        from langchain_community.vectorstores import Chroma
        db = Chroma.from_documents(all_chunks, embedding=embeddings)
        qa_chain = create_medical_qa_chain(db)

        uploaded_list = "<br>".join([f"📄 {name}" for name in uploaded_files])
        return "", [{"role": "assistant", "content": f"✅ Uploaded & indexed:<br>{uploaded_list}"}]

    except Exception as e:
        return "", [{"role": "assistant", "content": f"❌ Error: {str(e)}"}]

async def chat_interface(message, history):
    global fallback_chain

    history.append({"role": "user", "content": message})

    # Typing indicator
    typing_base = (
        f"<img src='{logo_base64}' style='width:24px; border-radius:50%; vertical-align: middle; margin-right: 8px;'>"
        " <em>AI-DOC is thinking{dots}</em>"
    )
    history.append({"role": "assistant", "content": typing_base.format(dots=".")})
    yield "", history

    # Simulate thinking animation
    for dots in ["..", "...", ""]:
        await asyncio.sleep(0.4)
        history[-1]["content"] = typing_base.format(dots=dots)
        yield "", history

    # Final response
    if qa_chain:
        result = answer_medical_question(message, qa_chain)
    else:
        if not fallback_chain:
            fallback_db = load_vectorstore()
            fallback_chain = create_medical_qa_chain(fallback_db)
        result = answer_medical_question(message, fallback_chain)

    history[-1] = {
        "role": "assistant",
        "content": f"<img src='{logo_base64}' style='width:28px; border-radius:50%; vertical-align: middle; margin-right: 8px;'>"
                   + result["answer"]
    }

    yield "", history

# Gradio UI
with gr.Blocks(css="""
#logo-container {
    justify-content: center;
    padding-top: 5px;
    margin-bottom: -30px;
}
#logo-img > div {
    background: linear-gradient(to bottom right, #2c2f36, #1f2125);
    box-shadow: none;
    padding: 0;
}
#title {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 0px;
}
.gradio-container {
    background: linear-gradient(to bottom right, #1e1e1e, #2c2c2c);
    color: #f0f0f0;
}

#upload-btn {
    height: 40px !important;
    width: 40px !important;
    min-width: 40px !important;
    border-radius: 50%;
    background-color: #3a3b45;
    text-align: center;
    padding: 0 !important;
    font-size: 18px !important;
    color: #a78bfa !important;
    border: none;
    margin-left: 4px;
}
#upload-btn:hover {
    background-color: #4c4d5a !important;
}

#hidden-file .wrap, #hidden-file .upload-box {
    display: none !important;
}

#subtitle {
    color: #f0f0f0;
    text-align: center;
    font-size: 1.2rem;
    margin-top: -15px;
    margin-bottom: -15px;
}
""", theme=gr.themes.Base()) as demo:

    with gr.Column():
        with gr.Row(elem_id="logo-container"):
            gr.Image(
                value="data/logo.png",
                show_label=False,
                show_download_button=False,
                width=90,
                height=90,
                elem_id="logo-img"
            )

        gr.HTML("<div id='title'>AI-DOC Medical Assistant 🩺</div>")
        gr.Markdown("AI personal assistant for medical insights!", elem_id="subtitle")

        chatbot = gr.Chatbot(
            value=[
                {"role": "assistant", "content":
                    f"<img src='{logo_base64}' style='width:28px; border-radius:50%; vertical-align: middle; margin-right: 8px;'>"
                    "Hi👋, I’m <strong>AI-DOC</strong> – your personal health assistant. Ask me anything!"
                }
            ],
            type="messages",
            height=380,
            show_copy_button=True,
        )

        with gr.Row():
            with gr.Column(scale=9):
                msg = gr.Textbox(placeholder="Ask your medical question...", show_label=False)
            with gr.Column(scale=1):
                upload_trigger = gr.Button("📎", elem_id="upload-btn")
                hidden_file = gr.File(file_types=[".pdf"], file_count="multiple", visible=True, elem_id="hidden-file")

        msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
        hidden_file.change(lambda f: upload_pdfs_inline(f) + [chatbot.value], [hidden_file], [msg, chatbot])
        upload_trigger.click(None, None, None, js="() => document.querySelector('#hidden-file input').click()")

if __name__ == "__main__":
    demo.launch()
