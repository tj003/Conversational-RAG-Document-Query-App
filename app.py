import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# set up streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload Pdf's and chat with their content")

## Input the GROQ api key
api_key = st.text_input("Enter your Groq API key:", type="password")

##check if groq api key is provided

if api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    #chat interface
    session_id = st.text_input("Session ID", value='default_session')

    ##stateful manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_file = st.file_uploader("Choose A pdf file", type="pdf", accept_multiple_files=True)

    ## Process uploaded PDF's
    if uploaded_file:
        documents = []
        for uploaded_file in uploaded_file:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

        # split adn create embeddings for the documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "Which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question 

        system_prompt = (
            "you are an assistant for questioon-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you dont't know the answer, say that you "
            " dont'tknow. Use three sentences maximum and keep the "
            "answer concise"
            " \n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_anser_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_anser_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()

            return st.session_state.store[session_id]
            
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, 
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input('Ask a question:')
        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke({'input': user_input}, config={'session_id': session_id})
            st.write('Session History:', session_history)
            st.write('Answer:', response['answer'])

else:
    st.warning("Please enter your Groq API key")
