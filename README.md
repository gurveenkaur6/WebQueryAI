# WebQueryAI 🕸️📑

WebQueryAI is an application that leverages LangChain framework and OpenAI's GPT-3.5-turbo to provide insightful answers to user queries based on the content of any loaded website. By integrating advanced text processing, vector embeddings, and retrieval-augmented generation (RAG) techniques, WebQueryAI offers an intelligent and efficient way to interact with web content.

## Features

- **Load Web Content**: Easily load and parse content from any URL.
- **Advanced Text Splitting**: Break down content into manageable chunks for better processing.
- **Vector Embedding and Retrieval**: Utilize sophisticated vector embedding techniques for accurate content retrieval.
- **Intelligent Question Answering**: Generate precise answers to questions using a powerful combination of prompt templates and the GPT-3.5-turbo language model through a Retrieval-Augmented Generation (RAG) chain.

## Tech Stack

### Backend

- **LangChain**: Provides the framework for creating and managing the various components used in the application, such as document loaders, text splitters, vector stores, and RAG chains.
- **OpenAI**: Utilizes the GPT-3.5-turbo model for generating high-quality responses.
- **BeautifulSoup**: Parses HTML content from web pages, focusing on specific elements like post content, titles, and headers.

### Frontend

- **Streamlit**: Offers a user-friendly interface for loading web content and interacting with the application. Streamlit handles user inputs, displays messages, and manages session states.

## Installation

### Prerequisites

- Python 3.8 or higher
- Streamlit
- BeautifulSoup
- LangChain
- OpenAI API Key

### Steps

1. Clone the repository:

    ```sh
    git clone https://github.com/your-username/WebQueryAI.git
    cd WebQueryAI
    ```

2. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Add your OpenAI API key:

    Create a `.env` file in the project root and add your OpenAI API key:

    ```sh
    OPENAI_API_KEY=your_openai_api_key
    ```

4. Run the Streamlit app:

    ```sh
    streamlit run webpage_qna.py
    ```

## Usage

1. Open the app in your web browser.
2. Enter a URL in the input field and click "Load URL".
3. Once the content is loaded, you can ask questions about the content in the chat input field.
4. The app will process your question and generate a response based on the content of the loaded webpage using the RAG approach.

