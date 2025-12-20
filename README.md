# RagInABox

The goal is to create a configurable "Rag In A Box" solution that can (1) given a list of folders, ingest the files from them and (2) given a list of websites, scrape the webpages and ingest the files (like PDFs) from those pages. The content will get chunked and put in a vector database (defaults to Azure AI Search). Finally, a Gradio frontend will hold the chat interface. The LLM will default to Azure OpenAI GPT5.2, embedding is text-embedding-3-small, and all keys, URLs, etc. are available via a .env.
