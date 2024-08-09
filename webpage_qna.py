import bs4  # to parse html content
import streamlit as st
import cohere
from langchain.prompts import PromptTemplate  # to define prompt templates manually
from langchain_community.document_loaders import WebBaseLoader  # doc loader for web-related content
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Vector store for document embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # OpenAI models for generating embeddings and responses
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

cohere_client = cohere.Client('7r8gktg66M4VjAFeBaKqLhU8A6yUWuh7Xn3IAFdI')

# Loading the Documents/webpage
def load_it(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")  # Target specific HTML classes
            )
        ),
    )
    return loader.load()

# to format the docs into a string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)  # Concatenate the content of all documents

# Function to rerank documents using Cohere's reranker
def rerank_documents_with_cohere(query, docs):
    documents = [doc.page_content for doc in docs]

    rerank_response = cohere_client.rerank(query=query, documents=documents, top_n=len(documents))

    # Extracting and formatting the rerank response details
    rerank_id = rerank_response.id
    results = rerank_response.results

    # Displaying the rerank response details
    print(f"\n--- Cohere Rerank Response ---")
    print(f"Query: {query}")
    print(f"Response ID: {rerank_id}")
    print(f"\n--- Results ---")

    for i, result in enumerate(results):
        doc_index = result.index
        relevance_score = result.relevance_score
        doc_content = docs[doc_index].page_content
        doc_source = docs[doc_index].metadata.get('source', 'No source available')

        print(f"\nDocument {i + 1}:")
        print(f"Relevance Score: {relevance_score:.6f}")
        print(f"Content:\n{doc_content}")
        print(f"Source: {doc_source}")
        print("-" * 80)  

    return [docs[result.index] for result in results]


# Function to generate a response based on the webpage content and user question
def generate_response(url, question):
    print(f"\n--- Generating Response ---")
    print(f"URL: {url}")
    print(f"Question: {question}")

    if url not in st.session_state:
        docs = load_it(url)
        # print(f"\nLoaded Documents:")
        # print(f"{docs}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split = text_splitter.split_documents(docs)
        # print(f"\nSplit Documents:")
        # print(f"{split}")

        vectorstore = Chroma.from_documents(documents=split, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()

        relevant_docs = retriever.get_relevant_documents(question)
        print(f"\nRelevant Documents:")
        print(f"{relevant_docs}")

        reranked_docs = rerank_documents_with_cohere(question, relevant_docs)
        print(f"\nReranked Documents:")
        print(f"{reranked_docs}")

        context = format_docs(reranked_docs)
        print(f"\nFormatted Context:")
        print(f"{context}")

        context_runnable = RunnableLambda(lambda _: {"context": context})

        # Define a custom prompt template
        prompt_template = PromptTemplate(
            template="You are an expert assistant who provides detailed and accurate explanations based on the given context and user query.\n\nContext:\n{context}\n\nQuestion:\n{question}",
            input_variables=["context", "question"]
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        rag_chain = (
            {"context": context_runnable, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )
        try:
            response = rag_chain.invoke({"question": question})
            print(f"\nGenerated Response:")
            print(f"{response}")
        except Exception as e:
            print(f"\nError during RAG chain invocation: {e}")
            return f"Error: {e}"
        return response

# Function to compare original results vs reranked results
def compare(query: str, top_k: int, top_n: int):
    print(f"\n--- Comparing Results ---")
    docs = load_it(st.session_state.website_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=split_docs, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    relevant_docs = retriever.get_relevant_documents(query)[:top_k]
    i2doc = {i: doc for i, doc in enumerate(relevant_docs)}

    rerank_response = cohere_client.rerank(
        query=query, documents=[doc.page_content for doc in relevant_docs], top_n=top_n
    )

    print(f"\nRerank Response for Comparison:")
    print(f"{rerank_response}")

    original_docs = []
    reranked_docs = []

    for i, result in enumerate(rerank_response.results):
        rerank_i = i2doc[i]
        if i != result.index:
            reranked_docs.append(f"[{result.index}] {result.document['text']}")
            original_docs.append(f"[{i}] {rerank_i.page_content}")

    for orig, rerank in zip(original_docs, reranked_docs):
        print(f"\n--- Comparison ---")
        print(f"ORIGINAL:\n{orig}")
        print(f"\nRERANKED:\n{rerank}")
        print(f"\n---\n")

## Streamlit app
st.title("Website QnA")
st.write("Enter any website URL and save time by asking questions based on the website content and more!")

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.docs = None
    st.session_state.website_url = ""

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
        st.session_state.messages.append({'role': 'user', 'content': question})
        try:
            response = generate_response(st.session_state.website_url, question)
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
        except Exception as e:
            st.error(f"An error occurred: {e}")

# compare(query="what is self reflection", top_k=5, top_n=5)
