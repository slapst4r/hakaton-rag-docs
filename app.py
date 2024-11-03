import streamlit as st
import emb_function
import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
from htmlt import css, bot_template, user_template
import pickle
import hashlib

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_user_id():
    import socket
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    return "1"


def save_session_state(file_path, file_name, user_id):
    if not os.path.exists(f"store_data/{user_id}/{file_path}"):
        os.mkdir(f"store_data/{user_id}/{file_path}")
    with open(f"store_data/{user_id}/{file_path}/{file_name}", 'wb') as f:
        pickle.dump(st.session_state.to_dict(), f)

def load_session_state(file_path, file_name, user_id):
    if not os.path.exists(f"store_data/{user_id}/"):
        os.mkdir(f"store_data/{user_id}/")

    if os.path.exists(f"store_data/{user_id}/{file_path}"):
        with open(f"store_data/{user_id}/{file_path}/{file_name}", 'rb') as f:
                loaded_state = pickle.load(f)
                for k, v in loaded_state.items():
                    st.session_state[k] = v
    else:
        print("Файл не найден")


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap = 100,
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


def open_pkl(user_id):
    if os.path.exists(f'store_data/{user_id}/test.pkl'):
        with open(f'store_data/{user_id}/test.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        return 0
    return data

def get_conversation_chain(vectorstore):
    llm = Ollama(model="ilyagusev/saiga_llama3:latest")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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
            
def get_avialable_sessions_for_user(user_id):
    user_folder = f"store_data/{user_id}/"
    if not os.path.exists(user_folder):
        return []  
    return [session for session in os.listdir(user_folder) if os.path.isdir(os.path.join(user_folder, session))]

def main():
    user_id = get_user_id()
    available_sessions = get_avialable_sessions_for_user(user_id)
    #st.write(user_id)
    st.set_page_config(page_title="RU RAG", page_icon=":ru:")
    st.write(css, unsafe_allow_html=True)
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("RAG NLP")
    st.subheader("Документы:")
    
    pdf_docs = st.file_uploader("Загрузите сюда файлы и нажмите 'Обработать'", accept_multiple_files=True)
    ses_name = st.text_input("Введите название сессии, если хотите снова к ней вернуться")
    #if ses_name != "": 
    
    #    save_session_state(ses_name, "test.pkl", get_user_id())
    if st.button('Обработать'):
            with st.spinner("Обработка..."):
                raw_text = get_pdf_text(pdf_docs)
                
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                #vectorstore = emb_function.get_embedding_function(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.ses_name = ses_name
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.user_id = user_id
                #save_session_state(ses_name, "test.pkl", get_user_id())
    user_question = st.text_input("Задайте вопрос о документах") 
    if user_question:
        handle_userinput(user_question)
        #save_session_state(ses_name, "test.pkl", get_user_id())
        #оно не работает НОРМАЛЬНООО
    if st.button("Сохранить сессию"):
        if ses_name != "":
            save_session_state(ses_name, 'test.pkl', user_id)
            st.success("Сессия успешно сохранена!")
        else:
            st.error('Невозможно сохранить сессию без имени')
    #st.write(user_template, unsafe_allow_html=True)
    #st.write(bot_template, unsafe_allow_html=True)
    
    with st.sidebar:
        
        st.subheader("Сесссии:")
        if available_sessions:
            selected_session = st.selectbox("Выберите сессию", options=available_sessions, index=None,)
            if selected_session == "":
                st.session_state = None
                pass
            load_session_state(selected_session, 'test.pkl', user_id)
        else:   
            st.write("Нет доступных сессий")
        st.write(st.session_state)
     
if __name__ == '__main__':
    main()