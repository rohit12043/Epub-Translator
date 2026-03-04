import os
import sys
import signal
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from dotenv import load_dotenv

from epub_processor import EPUBProcessor
from translator import GeminiTranslator
from multiprocessing import Process, Event, Queue
import multiprocessing

load_dotenv()

class EPUBTranslatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_gui()
        self.setup_variables()
        self.setup_signal_handlers()
        self.translation_process = None
        self.stop_event = None
        self.progress_queue = None

        # Components
        self.epub_processor = None
        self.translator = None
        
        # Translation state
        self.is_translating = False

    def setup_gui(self):
        """Setup the main GUI window and components"""
        self.root.title("EPUB Translator with Google AI Studio API")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Input EPUB
        ttk.Label(file_frame, text="Input EPUB:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.input_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.input_path_var, state="readonly").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10)
        )
        ttk.Button(file_frame, text="Browse", command=self.browse_input_file).grid(row=0, column=2)
        
        # Output EPUB
        ttk.Label(file_frame, text="Output Path:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.output_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_path_var).grid(
            row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(5, 0)
        )
        ttk.Button(file_frame, text="Browse", command=self.browse_output_file).grid(row=1, column=2, pady=(5, 0))
        
        row += 1
        
        # API Key frame
        api_frame = ttk.LabelFrame(main_frame, text="API Key", padding="5")
        api_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(api_frame, text="Google AI Studio API Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.api_key_var = tk.StringVar(value=os.environ.get('API_KEY'))
        ttk.Entry(api_frame, textvariable=self.api_key_var, show="*").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10)
        )
        api_frame.columnconfigure(1, weight=1)
        
        row += 1
        
        # Language settings frame
        lang_frame = ttk.LabelFrame(main_frame, text="Language Settings", padding="5")
        lang_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(lang_frame, text="Source Language:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.source_lang_var = tk.StringVar(value="Korean")
        source_combo = ttk.Combobox(lang_frame, textvariable=self.source_lang_var, state="readonly")
        source_combo['values'] = ("Korean", "English", "Japanese", "Chinese", "Spanish", "French", "German")
        source_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 20))
        
        ttk.Label(lang_frame, text="Target Language:").grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.target_lang_var = tk.StringVar(value="English")
        target_combo = ttk.Combobox(lang_frame, textvariable=self.target_lang_var, state="readonly")
        target_combo['values'] = ("English", "Spanish", "French", "German", "Japanese", "Korean", "Chinese")
        target_combo.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        lang_frame.columnconfigure(1, weight=1)
        lang_frame.columnconfigure(3, weight=1)
        
        row += 1
    
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Translation Settings", padding="5")
        settings_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(settings_frame, text="Max Translations per Session:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.max_translations_var = tk.StringVar(value="50")
        ttk.Entry(settings_frame, textvariable=self.max_translations_var, width=10).grid(row=0, column=1, sticky=tk.W)
        
        row += 1
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
        row += 1
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=(10, 0))
        
        self.start_button = ttk.Button(
            button_frame, text="Start Translation", command=self.start_translation
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self.stop_translation, state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="Exit", command=self.exit_application).pack(side=tk.LEFT)
        
        row += 1
        
        self.is_japanese_webnovel_var = tk.BooleanVar(value=False)
        jpn_checkbox = ttk.Checkbutton(
            settings_frame, 
            text="Is Japanese Webnovel (Force Japanese name transliteration)",
            variable=self.is_japanese_webnovel_var
        )
        jpn_checkbox.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        

    def setup_variables(self):
        """Setup internal variables"""
        from utils import setup_logging
        self.logger = setup_logging()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.root.after(0, self.exit_application)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def browse_input_file(self):
        """Browse for input EPUB file"""
        file_path = filedialog.askopenfilename(
            title="Select EPUB file",
            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
        )
        if file_path:
            self.input_path_var.set(file_path)
            # Auto-suggest output path
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}_translated.epub"
            self.output_path_var.set(str(output_path))

    def browse_output_file(self):
        """Browse for output EPUB file"""
        file_path = filedialog.asksaveasfilename(
            title="Save translated EPUB as",
            defaultextension=".epub",
            filetypes=[("EPUB files", "*.epub"), ("All files", "*.*")]
        )
        if file_path:
            self.output_path_var.set(file_path)

    def validate_inputs(self):
        """Validate user inputs"""
        if not self.input_path_var.get():
            messagebox.showerror("Error", "Please select an input EPUB file.")
            return False

        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please specify an output path.")
            return False

        if not os.path.exists(self.input_path_var.get()):
            messagebox.showerror("Error", "Input EPUB file does not exist.")
            return False

        from utils import validate_epub_file
        if not validate_epub_file(self.input_path_var.get()):
            messagebox.showerror("Error", "Invalid EPUB file.")
            return False

        if not self.api_key_var.get():
            messagebox.showerror("Error", "Please enter a valid Google AI Studio API key.")
            return False

        try:
            max_trans = int(self.max_translations_var.get())
            if max_trans <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Max translations must be a positive integer.")
            return False

        return True

    def start_translation(self):
        if not self.validate_inputs():
            return

        if self.translation_process and self.translation_process.is_alive():
            messagebox.showwarning("Warning", "Translation already running.")
            return

        self.stop_event = Event()
        self.progress_queue = Queue()

        self.translation_process = Process(
            target=translation_worker_process,
            args=(
                self.input_path_var.get(),
                self.output_path_var.get(),
                self.api_key_var.get(),
                self.source_lang_var.get(),
                self.target_lang_var.get(),
                int(self.max_translations_var.get()),
                self.is_japanese_webnovel_var.get(),
                self.stop_event,
                self.progress_queue
            )
        )

        self.translation_process.start()

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        self.root.after(100, self.check_progress_queue)

    def check_progress_queue(self):
        try:
            while not self.progress_queue.empty():
                msg_type, value = self.progress_queue.get()

                if msg_type == "progress":
                    self.progress_var.set(value)

                elif msg_type == "status":
                    self.status_var.set(value)

                elif msg_type == "done":
                    messagebox.showinfo("Success", value)
                    self.cleanup_after_process()
                    return

                elif msg_type == "stopped":
                    self.status_var.set(value)
                    self.cleanup_after_process()
                    return

                elif msg_type == "error":
                    messagebox.showerror("Error", value)
                    self.cleanup_after_process()
                    return

        except:
            pass

        if self.translation_process and self.translation_process.is_alive():
            self.root.after(100, self.check_progress_queue)
        else:
            self.cleanup_after_process()

    def cleanup_after_process(self):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

        if self.translation_process:
            self.translation_process.join(timeout=1)
            self.translation_process = None
                
    def stop_translation(self):
        if self.translation_process and self.translation_process.is_alive():
            self.stop_event.set()
            self.translation_process.join(timeout=3)

            if self.translation_process.is_alive():
                self.translation_process.terminate()

            self.cleanup_after_process()
            self.status_var.set("Translation stopped.")

    def update_progress(self, value):
        """Update progress bar"""
        self.root.after(0, lambda: self.progress_var.set(value))

    def update_status(self, message):
        """Update status label"""
        self.root.after(0, lambda: self.status_var.set(message))
        self.logger.info(message)

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.epub_processor:
                self.epub_processor.cleanup()
            if self.translator:
                self.translator.stop()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def exit_application(self):
        if self.translation_process and self.translation_process.is_alive():
            self.stop_event.set()
            self.translation_process.join(timeout=2)

            if self.translation_process.is_alive():
                self.translation_process.terminate()

        self.root.destroy()
            
    def run(self):
        """Run the GUI application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.exit_application)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.exit_application()

def translation_worker_process(
    input_path,
    output_path,
    api_key,
    source_lang,
    target_lang,
    max_translations,
    is_japanese_webnovel,
    stop_event,
    progress_queue
):
    try:
        epub_processor = EPUBProcessor()
        translator = GeminiTranslator(
            api_key=api_key,
            max_translations_per_session=max_translations,
            stop_event=stop_event
        )

        if not epub_processor.load_epub(input_path):
            progress_queue.put(("error", "Failed to load EPUB"))
            return

        def progress_callback(value):
            progress_queue.put(("progress", value))

        def status_callback(message):
            progress_queue.put(("status", message))

        success = epub_processor.translate_epub(
            translator,
            source_lang,
            target_lang,
            output_path,
            progress_callback=progress_callback,
            status_callback=status_callback,
            stop_flag=lambda: stop_event.is_set(),
            is_japanese_webnovel=is_japanese_webnovel,
        )

        if stop_event.is_set():
            translator.save_state()
            progress_queue.put(("stopped", "Translation stopped"))
        elif success:
            translator.save_state()
            progress_queue.put(("done", "Translation completed"))
        else:
            translator.save_state()
            progress_queue.put(("error", "Translation failed"))

    except Exception as e:
        progress_queue.put(("error", str(e)))
        
def main():
    """Main entry point"""
    app = EPUBTranslatorGUI()
    app.run()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()