import streamlit as st
import emb_function
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
from htmlt import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(
        model="all-minilm:l6-v2"
    )
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = Ollama(model="ilyagusev/saiga_llama3:latest", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_hist", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="RU RAG", page_icon=":ru:")
    st.write(css, unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    st.header("Файлики")
    user_question = st.text_input("Задайте вопрос о документах")
    if user_question:
        handle_userinput(user_question)
    st.write(user_template, unsafe_allow_html=True)
    st.write(bot_template, unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Документы:")
        pdf_docs = st.file_uploader("Загрузите сюда файлы и нажмите 'Обработать'", accept_multiple_files=True)
        if st.button('Обработать'):
            with st.spinner("Обработка..."):
                raw_text = get_pdf_text(pdf_docs)
                
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                #vectorstore = emb_function.get_embedding_function(text_chunks)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        
if __name__ == '__main__':
    main()