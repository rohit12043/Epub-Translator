# EPUB Translator (v3.0 Revamp)
EPUB Translator is a Python-based application designed to make written content more accessible by translating EPUB-format books between languages utilizing the official Google Gemini API. The application provides a simple graphical user interface (GUI) and a modular backend to handle end-to-end translation of EPUB content while preserving original HTML formatting, document structure, and metadata.

**Version 3.0** represents a significant revamp, introducing **Adaptive Scheduling**, **HTML Tag Preservation**, and **Batched NER** for a more robust, high-quality translation experience.

## Workflow
<img width="778" height="408" alt="image" src="https://github.com/user-attachments/assets/d8f0ee54-8d38-42ec-912c-12f3b28fb02c" />

The following translation loop defines the core logic of the application:
1. **Parse:** The app extracts text from XHTML files using `ebooklib` and `BeautifulSoup`.
2. **Analyze:** The **Batched NER engine** identifies proper nouns (characters, organizations) in a single pass to build a consistent glossary.
3. **Preserve:** The **TagPreserver** replaces inline HTML tags (em, strong, ruby, etc.) with tokens (`[T0]`, `[T1]`) to protect them during translation.
4. **Translate:** Chunks are sent to the best available Gemini model via the **Adaptive Scheduler**, which manages rate limits (RPM, TPM, RPD) dynamically.
5. **Restore & Compile:** Tokens are swapped back to their original tags, and the translated content is repacked into a new EPUB.

## Core Features

* **HTML Tag Preservation (TagPreserver):** Guarantees that inline formatting (like *italics*, **bold**, `<ruby>` tags, and custom `<span>` styles) survives translation perfectly by shielding them from the LLM using a tokenization-restoration system.
* **Adaptive Scheduling & Multi-Model Fallback:** Dynamically prioritizes models (e.g., `gemini-2.0-flash`, `gemini-1.5-pro`) based on real-time capacity. Automatically handles rate-limit cooldowns and waits for availability.
* **Batched Named Entity Recognition (NER):** Performs an efficient initial AI pass to identify proper nouns, ensuring consistent character names and terminology throughout the book via a local JSON glossary.
* **Intelligent Text Chunking:** Splits text at natural boundaries (paragraphs, sentences) using mathematical token estimation to stay within API limits while maintaining narrative context.
* **Persistent Checkpointing:** Saves progress at the chunk level in `checkpoints/`, allowing you to resume interrupted translations exactly where they left off.
* **Advanced Error Handling:** Implements exponential backoff and adaptive delays to gracefully handle API congestion and transient errors.

## Supported Languages
The application currently supports translation between the following languages:

* English
* Korean
* Japanese
* Chinese
* Spanish
* French
* German

*Note: The translation prompts in `prompts.py` are currently optimized for narrative coherence in light novel and web novel formats.*

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
* **Japanese Webnovel Toggle:** Enables a specialized NER mode to prioritize Japanese Romaji transliteration over Chinese Pinyin.
6. **Execution:** Click "Start Translation". Progress is saved automatically to `checkpoints/`.

## Logging and Debugging
All application actions, rate limit adjustments, errors, and validation flags are written to `logs/translator.log`. Detailed model usage and glossary data are maintained in the `checkpoints/` directory.

## Contributing
Contributions to improve the application are welcome. To contribute:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/FeatureName`).
3. Commit your changes (`git commit -m 'Add FeatureName'`).
4. Push to the branch (`git push origin feature/FeatureName`).
5. Open a Pull Request with a detailed description of the changes.

## Disclaimer
This project is in an early stage and is a personal/hobby project. While it handles most common EPUB structures, unexpected issues might occur depending on the complexity of your EPUB's formatting or Gemini API behavior.

There’s no fixed roadmap or guaranteed updates. If you find bugs or want to suggest improvements, feel free to open an issue or fork the project.
