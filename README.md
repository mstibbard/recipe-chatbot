# Recipe Chatbot

> 📝 **Note:** This project serves as a foundation for ongoing development throughout the AI Evals course. We will be incrementally adding features and refining its capabilities in subsequent lessons and homework assignments.

This project provides a starting point for building and evaluating an AI-powered Recipe Chatbot. You will be working with a web application that uses FastAPI for the backend and a simple HTML/CSS/JavaScript frontend. The core of the chatbot involves interacting with a Large Language Model (LLM) via LiteLLM to get recipe recommendations.

Your main tasks will be to refine the chatbot's persona and intelligence by crafting a detailed system prompt, expanding its test query dataset, and evaluating its performance.

![Recipe Chatbot UI](./screenshots/hw1.png)

## Table of Contents

- [Core Components Provided](#core-components-provided)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Provided Application](#running-the-provided-application)
  - [1. Run the Web Application (Frontend and Backend)](#1-run-the-web-application-frontend-and-backend)
  - [2. Run the Bulk Test Script](#2-run-the-bulk-test-script)
- [Homework Assignment 1: Write a Starting Prompt](#homework-assignment-1-write-a-starting-prompt)

## Core Components Provided

This initial setup includes:

*   **Backend (FastAPI)**: Serves the frontend and provides an API endpoint (`/chat`) for the chatbot logic.
*   **Frontend (HTML/CSS/JS)**: A basic, modern chat interface where users can send messages and receive responses.
    *   Renders assistant responses as Markdown.
    *   Includes a typing indicator for better user experience.
*   **LLM Integration (LiteLLM)**: The backend connects to an LLM (configurable via `.env`) to generate recipe advice.
*   **Bulk Testing Script**: A Python script (`scripts/bulk_test.py`) to send multiple predefined queries (from `data/sample_queries.csv`) to the chatbot's core logic and save the responses for evaluation. This script uses `rich` for pretty console output.

## Project Structure

```
recipe-chatbot/
├── backend/
│   ├── __init__.py
│   ├── main.py         # FastAPI application, routes
│   └── utils.py        # LiteLLM wrapper, system prompt, env loading
├── data/
│   └── sample_queries.csv # Sample queries for bulk testing (ID, Query)
├── frontend/
│   └── index.html      # Chat UI (HTML, CSS, JavaScript)
├── results/            # Output folder for bulk_test.py
├── scripts/
│   └── bulk_test.py    # Bulk testing script
├── .env.example        # Example environment file
├── env.example         # Backup env example (can be removed if .env.example is preferred)
├── pyproject.toml      # Python dependencies
└── README.md           # This file (Your guide!)
```

## Setup Instructions

1.  **Clone the Repository (if you haven't already)**
    ```bash
    git clone https://github.com/ai-evals-course/recipe-chatbot.git
    cd recipe-chatbot
    ```

2. Install dependencies
    ```bash
    # Create venv and install dependencies
    uv sync

    # Activate the virtual env
    source ./venv/bin/activate
    ```

3.  **Configure Environment Variables (`.env` file)**
    *   Copy the example environment file:
        ```bash
        cp env.example .env
        ```
        (or `cp .env.example .env` if you have that one)
    *   Edit the `.env` file. You will need to:
        1.  Set the `MODEL_NAME` to the specific model you want to use (e.g., `openai/gpt-3.5-turbo`, `anthropic/claude-3-opus-20240229`, `ollama/llama2`).
        2.  Set the **appropriate API key environment variable** for the chosen model provider.
            Refer to your `env.example` for common API key names like `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.
            LiteLLM will automatically use these provider-specific keys.

        Example of a configured `.env` file if using an OpenAI model:
        ```env
        MODEL_NAME=openai/gpt-3.5-turbo
        OPENAI_API_KEY=sk-yourActualOpenAIKey...
        ```
        Example for an Anthropic model:
        ```env
        MODEL_NAME=anthropic/claude-3-haiku-20240307
        ANTHROPIC_API_KEY=sk-ant-yourActualAnthropicKey...
        ```

    *   **Important - Model Naming and API Keys with LiteLLM**:
        LiteLLM supports a wide array of model providers. To use a model from a specific provider, you generally need to:
        *   **Prefix the `MODEL_NAME`** correctly (e.g., `openai/`, `anthropic/`, `mistral/`, `ollama/`).
        *   **Set the corresponding API key variable** in your `.env` file (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`). Some local models like Ollama might not require an API key.

        Please refer to the official LiteLLM documentation for the correct model prefixes and required environment variables for your chosen provider: [LiteLLM Supported Providers](https://docs.litellm.ai/docs/providers).

## Running the Provided Application

### 1. Run the Web Application (Frontend and Backend)

*   Ensure your virtual environment is activated and your `.env` file is configured.
*   From the project root directory, start the FastAPI server using Uvicorn:
    ```bash
    uv run uvicorn backend.main:app --reload
    ```
*   Open your web browser and navigate to: `http://127.0.0.1:8000`

    You should see the chat interface.


### 2. Run the Bulk Test Script

The bulk test script allows you to evaluate your chatbot's responses to a predefined set of queries. It sends queries from `data/sample_queries.csv` directly to the backend agent logic and saves the responses to the `results/` directory.

*   Ensure your virtual environment is activated and your `.env` file is configured.
*   From the project root directory, run:
    ```bash
    uv run scripts/bulk_test.py
    ```
*   To use a different CSV file for queries:
    ```bash
    uv run scripts/bulk_test.py --csv path/to/your/queries.csv
    ```
    The CSV file must have `id` and `query` columns.
*   Check the `results/` folder for a new CSV file containing the IDs, queries, and their corresponding responses. This will be crucial for evaluating your system prompt changes.
