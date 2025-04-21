import gradio as gr
from app.core.document_processor import chunk_documents, get_embeddings
from app.core.retrieval import load_vectorstore
from app.core.groq_client import create_medical_qa_chain, answer_medical_question
from langchain_community.document_loaders import PyPDFLoader
import base64

# Global chain state
qa_chain = None
fallback_chain = None

def encode_image_to_base64(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"
    
def upload_pdfs(file_paths):
    global qa_chain
    if not file_paths:
        return "<span style='color: red;'>❌ No files selected.</span>"

    try:
        all_chunks = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)

        embeddings = get_embeddings()

        from langchain_community.vectorstores import Chroma
        db = Chroma.from_documents(all_chunks, embedding=embeddings)
        qa_chain = create_medical_qa_chain(db)

        return f"<span style='color: green;'>✅ {len(file_paths)} file(s) uploaded and indexed successfully!</span>"
    except Exception as e:
        return f"<span style='color: red;'>❌ Error: {str(e)}</span>"
    
logo_base64 = encode_image_to_base64("data/logo.png")

def chat_interface(message, history):
    global fallback_chain
    if qa_chain:
        result = answer_medical_question(message, qa_chain)
    else:
        if not fallback_chain:
            fallback_db = load_vectorstore()
            fallback_chain = create_medical_qa_chain(fallback_db)
        result = answer_medical_question(message, fallback_chain)
    bot_response = (
        f"<img src='{logo_base64}' style='width:40px; border-radius:50%; vertical-align: middle; margin-right: 8px;'>"
        + result["answer"]
    )
    return bot_response


# Gradio UI
with gr.Blocks(css="""
    #logo-container {
        justify-content: center;
        padding-top: 10px;
        margin-bottom: -30px;
    }

    #logo-img > div {
        background: linear-gradient(to bottom right, #e3f2fd, #ebeef9);
        box-shadow: none;
        padding: 0;
    }
    #title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2b4f81;
        margin-bottom: 20px;
    }
    .custom-upload .wrap, 
    .custom-upload .upload-box {
        min-height: 100px;
        max-height: 140px;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .custom-upload .upload-box svg {
        height: 60px;
        width: 20px;
    }
    .custom-upload .upload-box label {
        font-size: 14px;
        line-height: 1.2;
    }
    .gradio-container {
        background: linear-gradient(to bottom right, #e3f2fd, #f3e5f5);
    }
               
""", theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue")) as demo:

    with gr.Column():
        with gr.Row(elem_id="logo-container"):
            gr.Image(
                value="data/logo.png",  # Make sure this is the correct path relative to where your script runs
                show_label=False,
                show_download_button=False,
                width=150,
                height=150,
                elem_id="logo-img"
            )
        gr.HTML("<div id='title'>AI-DOC Medical Assistant 🩺</div>")
        gr.Markdown("Upload one or more medical PDFs to ask questions. If not, AI-DOC will use Merck Manual by default.")

        file_input = gr.File(label="📄 Upload Medical PDFs", file_types=[".pdf"], type="filepath", file_count="multiple",elem_id="file-upload",scale=1,elem_classes="custom-upload")
        upload_status = gr.Markdown()
        file_input.change(fn=upload_pdfs, inputs=[file_input], outputs=[upload_status])

        gr.ChatInterface(
            fn=chat_interface,
            title="AI-DOC Chatbot",
            chatbot=gr.Chatbot(
                value=[("",f"<img src='{logo_base64}' style='width:40px; border-radius:50%; vertical-align: middle; margin-right: 8px;'>""👋 Hi, I’m AI-DOC – your personal health assistant. How can I help you today?")]
            )
        )

# Launch the app
if __name__ == "__main__":
    demo.launch()
