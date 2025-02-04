# Wikipedia RAG-Bot 🤖

A LangChain-based question-answering chatbot that uses RAG (Retrieval-Augmented Generation) to answer queries about Wikipedia articles. The bot retrieves relevant information from Wikipedia articles and provides concise, accurate answers using OpenAI's GPT-4o-mini model.

## Features

- Interactive Wikipedia topic search
- Real-time article retrieval and processing
- RAG-powered question answering
- Streamlit web interface
- Document chunking and semantic search
- Memory-based conversation tracking
- LangSmith integration for monitoring

## Prerequisites

- Python 3.8+
- OpenAI API key
- LangSmith API key (for tracing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd wikipedia-rag-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```plaintext
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
```

## Dependencies

- `streamlit`: Web interface
- `langchain`: LLM framework
- `openai`: OpenAI API integration
- `python-dotenv`: Environment variable management
- `wikipedia`: Wikipedia article retrieval
- `langgraph`: Graph-based LLM orchestration

## Usage

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Enter a Wikipedia topic in the first input field
3. Once the article is loaded, enter your question in the second input field
4. The bot will retrieve relevant information and provide a concise answer

## How It Works

1. **Article Retrieval**: Uses `WikipediaLoader` to fetch articles based on user input
2. **Document Processing**: 
   - Splits articles into manageable chunks using `RecursiveCharacterTextSplitter`
   - Creates embeddings using OpenAI's embedding model
   - Stores chunks in an in-memory vector store

3. **Query Processing**:
   - Uses a graph-based approach with three main nodes:
     - `query_or_respond`: Generates tool calls for retrieval
     - `tools`: Executes retrieval operations
     - `generate`: Produces final responses

4. **Response Generation**:
   - Retrieves relevant chunks using semantic search
   - Generates concise answers (maximum three sentences)
   - Handles cases where information is not available

## Configuration

The application uses several configurable parameters:

- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters
- `max_docs`: 1 (number of Wikipedia articles to retrieve)
- `k`: 2 (number of chunks to retrieve per query)

## Limitations

- Currently only processes one Wikipedia article at a time
- Responses are limited to three sentences
- Requires active internet connection for Wikipedia access
- API keys must be properly configured

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://opensource.org/licenses/MIT) file for details.
