import streamlit as st
from transformers import pipeline
from huggingface_hub import login
summarizer = pipeline("summarization", model="MK-5/t5-small-Abstractive-Summarizer")
st.title("Abstractive Text Summarizer")
st.write("This application summarizes text using a trained T5 transformer model.")
option = st.radio("Choose input method:", ("Enter text", "Upload a file"))
if option == "Enter text":
    user_input = st.text_area("Enter text to summarize:", height=300)
else:
    uploaded_file = st.file_uploader("Upload a text file (txt, pdf):", type=["txt", "pdf"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "txt":
            user_input = uploaded_file.read().decode("utf-8")
        elif file_extension == "pdf":
            import fitz 
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            user_input = ""
            for page in pdf_document:
                user_input += page.get_text()
            pdf_document.close()
        else:
            st.error("Unsupported file type. Please upload a .txt or .pdf file.")
            user_input = None
summary_length = st.slider("Select maximum summary length (in words):", min_value=30, max_value=200, value=50)
if st.button("Summarize"):
    if user_input:
        with st.spinner("Summarizing..."):
            summary = summarizer(user_input, max_length=summary_length, min_length=30, do_sample=False)[0]["summary_text"]
        st.subheader("Summary")
        st.write(summary)
    else:
        st.warning("Please enter text or upload a valid file.")
st.markdown("---")
st.markdown("Developed by [Ronit Debnath](https://github.com/Masterkrusher5).")
