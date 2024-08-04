import bs4 # to parse html content
import streamlit as st 
from langchain import hub # to use hub for reusable prompt templates
from langchain_community.document_loaders import WebBaseLoader # doc loader for web related content
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma # Vector store for document embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI models for generating embeddings and responses
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Loading the Documents/webpage
def load_it(url):
    # Initialize the web page loader with the given URL and BeautifulSoup settings
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")  # Target specific HTML classes
            )
        ),
    )
    # load and return the content 
    return loader.load()

# to format the docs into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) # Concatenate the content of all documents

# Function to generate a response based on the webpage content and user question
def generate_response(url, question):

    # Check if the URL is not already processed in the session state
    if url not in st.session_state:
        docs = load_it(url)  # Load the webpage content
        # print(f"Loaded documents: {docs}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split = text_splitter.split_documents(docs) # Split the documents into smaller chunks
        # print(f"Split documents: {split}")

        # Create a vector store from the document chunks using OpenAI embeddings
        vectorstore= Chroma.from_documents(documents=split, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever() # Initialize a retriever from the vector store
        print(f"Retriever initialized: {retriever}")


        # Pull a prompt template from the LangChain hub
        prompt_template = hub.pull("rlm/rag-prompt")
        print(f"Prompt template: {prompt_template}")

        # Initialize the language model for generating responses
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        print(f"Language model initialized: {llm}")

        
        # Retrieve relevant documents using the question as the query
        relevant_docs = retriever.get_relevant_documents(question)

        # Format the relevant documents into a string
        context = format_docs(relevant_docs)
        print(f"context retrieved:{context}")
        # Create a Runnable for context
        context_runnable = RunnableLambda(lambda _: {"context": context})

        # Define the RAG chain
        rag_chain = (
            {"context": context_runnable, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        try:
            # Invoke the RAG chain with the user question to generate a response
            response = rag_chain.invoke({"question": question})
        except Exception as e:
            print(f"Error during RAG chain invocation: {e}")
            return f"Error: {e}"
        return response

## Streamlit app
st.title("Website QnA")
st.write("Enter any website URL and save time by asking questions based on the website content and more!")

# Initialize session state variables if not already set
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.docs = None
    st.session_state.website_url = ""

# URL input only appears if no URL has been loaded yet
if st.session_state.docs is None:
    website_url = st.text_input("Enter the website URL", key="website_url_input")
    if st.button("Load URL"):
        if website_url:
            with st.spinner("Loading and processing the webpage..."):
                try:
                    st.session_state.docs = load_it(website_url)
                    st.session_state.website_url = website_url
                    st.success("Website content loaded! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.write(f"Loaded URL: {st.session_state.website_url}")
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    question = st.chat_input("Enter your query")
    if question:
        st.chat_message('user').markdown(question)
        st.session_state.messages.append({'role': 'user', 'content': question})  # Use question as a string
        try:
            response = generate_response(st.session_state.website_url, question)  # Pass question as a string
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
        except Exception as e:
            st.error(f"An error occurred: {e}")

