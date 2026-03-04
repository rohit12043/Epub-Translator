import os
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET
def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger() # Get root logger
    logger.setLevel(log_level)
    
    # Prevent adding handlers multiple times if called twice
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / "translator.log", encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logging.getLogger("EPUBTranslator")

def validate_epub_file(filepath: str) -> bool:
    """Validate if file is a proper EPUB"""
    try:
        if not os.path.exists(filepath):
            return False
        if not filepath.lower().endswith('.epub'):
            return False
        if not zipfile.is_zipfile(filepath):
            return False
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            if 'mimetype' not in zip_file.namelist():
                return False
            mimetype_content = zip_file.read('mimetype').decode('utf-8').strip()
            if mimetype_content != 'application/epub+zip':
                return False
            meta_inf_files = [f for f in zip_file.namelist() if f.startswith('META-INF/')]
            if not meta_inf_files:
                return False
            if 'META-INF/container.xml' not in zip_file.namelist():
                return False
        return True
    except Exception as e:
        logging.getLogger("EPUBTranslator").error(f"Error validating EPUB file: {e}")
        return False

def get_epub_metadata(filepath: str) -> Dict[str, Any]:
    """Extract basic metadata from EPUB file"""
    metadata = {'title': 'Unknown', 'author': 'Unknown', 'language': 'Unknown', 'chapters': 0, 'file_size': 0}
    try:
        if not validate_epub_file(filepath):
            return metadata
        metadata['file_size'] = os.path.getsize(filepath)
        with zipfile.ZipFile(filepath, 'r') as zip_file:
            container_content = zip_file.read('META-INF/container.xml').decode('utf-8')
            container_root = ET.fromstring(container_content)
            opf_path = None
            for rootfile in container_root.findall('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile'):
                opf_path = rootfile.get('full-path')
                break
            if opf_path:
                opf_content = zip_file.read(opf_path).decode('utf-8')
                opf_root = ET.fromstring(opf_content)
                title_elem = opf_root.find('.//{http://purl.org/dc/elements/1.1/}title')
                if title_elem is not None and title_elem.text:
                    metadata['title'] = title_elem.text
                author_elem = opf_root.find('.//{http://purl.org/dc/elements/1.1/}creator')
                if author_elem is not None and author_elem.text:
                    metadata['author'] = author_elem.text
                lang_elem = opf_root.find('.//{http://purl.org/dc/elements/1.1/}language')
                if lang_elem is not None and lang_elem.text:
                    metadata['language'] = lang_elem.text
                html_files = [
                    f for f in zip_file.namelist()
                    if f.endswith(('.html', '.xhtml', '.htm'))
                    and not any(skip in f.lower() for skip in ['toc', 'nav', 'cover', 'title', 'copyright', 'index', 'info'])
                ]
                metadata['chapters'] = len(html_files)
    except Exception as e:
        logging.getLogger("EPUBTranslator").error(f"Error extracting EPUB metadata: {e}")
    return metadata

def clean_temp_files(base_path: str) -> None:
    try:
        temp_epub = Path(base_path).with_suffix('.temp.epub')
        if temp_epub.exists():
            temp_epub.unlink()
            logging.getLogger("EPUBTranslator").info(f"Cleaned temp file: {temp_epub}")
    except Exception as e:
        logging.getLogger("EPUBTranslator").error(f"Error cleaning temp files: {e}")