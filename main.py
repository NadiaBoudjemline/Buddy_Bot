# pip install streamlit PyPDF2 langchain langchain_core langchain_openai python-dotenv
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage

def main():
    load_dotenv()
    st.set_page_config(page_title="BuddyBot", page_icon="🤖")
    st.header("BuddyBot 🤖")

    PDF = PdfReader("Banque.pdf")
    #Extract the text from the pdf
    text = ""
    for page in PDF.pages:
        text+=page.extract_text()
    
    #Split content(text) into chunks
    Text_spliter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 200,
        chunk_overlap = 50,
        length_function = len,
    )
    Chunks = Text_spliter.split_text(text)

    #Create embeddings
    Embeddings = OpenAIEmbeddings()
    Document = FAISS.from_texts(Chunks, Embeddings)
    
    #User input
    user_question = st.chat_input("Ask me a question")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content = "Hello, I am a bot.How can I help you?"),
                                         ] 
    if user_question:
        docs = Document.similarity_search(user_question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = docs, question = user_question)
        st.session_state.chat_history.append(HumanMessage(content = user_question))
        st.session_state.chat_history.append(AIMessage(content = response))

    #Conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
   

if __name__ == '__main__':
    main()