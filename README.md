# EPUB Translator
EPUB Translator is a Python-based application designed to make written content more accessible by translating EPUB-format books between languages utilizing the official Google Gemini API. The application provides a simple graphical user interface (GUI) and a modular backend to handle end-to-end translation of EPUB content while preserving original HTML formatting, document structure, and metadata.

Version 2.0 represents a complete architectural rewrite, transitioning from legacy browser automation to a robust, API-driven backend. Key improvements include multi-model support, dynamic rate limiting, automated Named Entity Recognition (NER), and persistent checkpointing for resumable translation sessions.

## Workflow
<img width="778" height="408" alt="image" src="https://github.com/user-attachments/assets/d8f0ee54-8d38-42ec-912c-12f3b28fb02c" />

The following translation loop defines the core logic of the application:
1. **Parse:** The app extracts text from XHTML files.
2. **Analyze:** The NER engine builds a translation map for proper nouns.
3. **Process:** The engine chunks text based on API token limits.
4. **Translate:** Chunks are sent to Gemini, with automatic fallback for rate limits.
5. **Compile:** Translated text is mapped back to original HTML and saved to the final EPUB.

### Frontend
<img width="500" alt="image" src="https://github.com/user-attachments/assets/64554b16-0052-4d7f-a854-add0a4fc33e9" />

**GUI:** Built with Tkinter, the interface operates asynchronously from the translation engine using isolated multiprocessing. This ensures the UI remains fully responsive while the backend handles heavy API requests, rate limit cooldowns, and chunking.

## Core Features

* **Format Preservation:** Translates textual content while maintaining EPUB styling, tags, and structure via `BeautifulSoup` and a custom DOM reconstruction algorithm.
* **Multi-Model Fallback:** Interfaces with multiple Gemini models (e.g., `gemini-2.5-flash` for speed, `gemini-3.1-pro` for quality), automatically falling back to alternatives if rate limits are reached.
* **Automated NER:** Performs an initial AI pass to identify and translate proper nouns. These are stored in a local JSON cache and replaced with placeholders during the main translation pass to enforce consistency automatically.
* **Persistent Checkpointing:** Saves translation state at the HTML chunk level to a local JSON file, allowing sessions to be interrupted and resumed without data loss.
* **Intelligent Chunking:** Splits text at safe boundaries using mathematical token estimation to strictly adhere to API input limits and preserve narrative coherence.

## Supported Languages
The application currently supports translation between the following languages:

* English
* Korean
* Japanese
* Chinese
* Spanish
* French
* German

*Note: The translation prompts in `prompts.py` are currently optimized for narrative coherence in light novel and web novel formats, but can be modified for general use or other language pairs.*

## Requirements
### System Requirements

* Python 3.8 or higher
* Google account with access to AI Studio and a valid API key
* Internet connection

### Python Dependencies
Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The application relies on the following core Python packages:

* `ebooklib` (EPUB parsing and manipulation)
* `google-genai` (Official SDK for the Google Gemini API)
* `beautifulsoup4`, `lxml` (HTML DOM parsing)
* `tkinter` (Standard GUI library)
* `filelock`, `python-dotenv` (File handling and environment management)

## Setup Instructions
1. **Clone the Repository**
```bash
git clone https://github.com/rohit12043/epub-translator.git
cd epub-translator
```


2. **Install Dependencies**
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```


3. **Configure API Key**
* Go to [Google AI Studio](https://aistudio.google.com/) and sign in.
* Generate a Gemini API key.
* Create a `.env` file in the project root directory and add the key:


```text
API_KEY=your_api_key_here
```

## Usage Guide
### Launching the Application
Run the main script to initialize the GUI:

```bash
python main.py
```

### Interface Configuration
1. **Select Input File:** Click "Browse" to choose the source EPUB file.
2. **Set Output Path:** Specify the destination directory and filename for the translated EPUB.
3. **Language Selection:** Define the source language and the desired target language.
4. **API Key:** The application automatically loads the key from the `.env` file or direct input.
5. **Advanced Configurations:**
* **Japanese Webnovel Toggle:** Enables a specialized NER mode to prioritize Japanese Romaji transliteration over Chinese Pinyin when translating from Chinese-translated Japanese sources.
6. **Execution:** Click "Start Translation" to begin. Use the "Stop" or "Pause" buttons to safely halt the process and create a checkpoint.

## Logging and Debugging
All application actions, rate limit adjustments, errors, and validation flags are written to `logs/translator.log`. Review this file for auditing purposes or to troubleshoot prolonged pauses caused by API rate limits.

## Contributing
Contributions to improve the application are welcome. To contribute:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/FeatureName`).
3. Commit your changes (`git commit -m 'Add FeatureName'`).
4. Push to the branch (`git push origin feature/FeatureName`).
5. Open a Pull Request with a detailed description of the changes.

## Disclaimer
This project is in a very early stage and is a personal/hobby project. It's been tested on a few books, but may still contain bugs or rough edges. While most common use cases should work fine, unexpected issues might come up depending on the structure of your EPUB or how the Gemini API behaves.

There’s no fixed roadmap or guaranteed updates. I may continue to improve it when I get time, or leave it as-is. If you find bugs or want to suggest improvements, feel free to open an issue or fork the project.
