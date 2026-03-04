
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TrieNode:
    """Node for glossary term matching trie."""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.term = None
        self.target = None

class GlossaryManager:
    """Manages loading, saving, and querying a translation glossary."""
    def __init__(self, glossary_file: str = "glossary.json"):
        self.glossary_file = glossary_file
        self.glossary: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
        self._load_glossary()
        self.trie = self.build_trie()

    def _load_glossary(self) -> None:
        """Loads glossary from file with validation."""
        try:
            if os.path.exists(self.glossary_file):
                with open(self.glossary_file, 'r', encoding='utf-8') as f:
                    raw_glossary = json.load(f)
                    if isinstance(raw_glossary, dict):
                        self.glossary = {k: v for k, v in raw_glossary.items() if isinstance(k, str) and isinstance(v, str)}
                    else:
                        self.logger.warning(f"Invalid glossary format in {self.glossary_file} (not a dict), initializing empty.")
                        self.glossary = {}
                self.logger.info(f"Loaded {len(self.glossary)} glossary entries from {self.glossary_file}.")
            else:
                self.logger.info(f"No glossary file found at {self.glossary_file}, starting with empty glossary.")
                self.glossary = {}
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse glossary JSON from {self.glossary_file}. Initializing empty glossary.")
            self.glossary = {}
        except Exception as e:
            self.logger.error(f"Failed to load glossary from {self.glossary_file}: {e}. Initializing empty glossary.")
            self.glossary = {}

    def save_glossary(self) -> None:
            try:
                file_path = Path(self.glossary_file)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                temp_file = file_path.with_suffix('.json.tmp')
                with temp_file.open('w', encoding='utf-8') as f:
                    json.dump(self.glossary, f, indent=2, ensure_ascii=False)
                
                temp_file.replace(file_path)
                self.logger.debug(f"Glossary saved: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save glossary: {e}")

    def add_entry(self, source: str, target: str, force_update: bool = False) -> bool:
        """Adds or updates a glossary entry. Returns True if added/updated, False otherwise."""
        if not source or not target or not isinstance(source, str) or not isinstance(target, str):
            self.logger.warning(f"Attempted to add invalid glossary entry: source='{source}', target='{target}'")
            return False

        normalized_source = source.strip()
        normalized_target = target.strip()

        if not normalized_source or not normalized_target:
            self.logger.warning(f"Attempted to add glossary entry with empty normalized source or target: '{source}' -> '{target}'")
            return False

        if normalized_source.lower() == normalized_target.lower():
            self.logger.debug(f"Skipping glossary entry where source and target are effectively the same: '{source}' -> '{target}'")
            return False

        if source in self.glossary and not force_update:
            if self.glossary[source] != target:
                self.logger.debug(f"Glossary entry '{source}' already exists with different target '{self.glossary[source]}'. Not updated (force_update=False).")
            else:
                self.logger.debug(f"Glossary entry '{source}' already exists with same target '{target}'. No update needed.")
            return False

        self.glossary[source] = target
        self.save_glossary()
        self.trie = self.build_trie()
        action = "Updated" if force_update and source in self.glossary else "Added"
        self.logger.info(f"{action} glossary entry: '{source}' -> '{target}'. Trie rebuilt.")
        return True

    def build_trie(self) -> TrieNode:
        """Builds a case-sensitive trie for term matching from the current glossary."""
        root = TrieNode()
        sorted_terms = sorted(self.glossary.keys(), key=len, reverse=True)
        for term in sorted_terms:
            if not term:
                continue
            node = root
            for char in term:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.term = term
            node.target = self.glossary[term]
        self.logger.debug(f"Trie built with {len(self.glossary)} terms.")
        return root

    def find_terms_in_text(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Finds all non-overlapping glossary terms in text using the trie.
        Prioritizes the longest match at any given start position due to sorted terms in trie build.
        """
        matches = []
        n = len(text)
        i = 0
        while i < n:
            node = self.trie
            longest_match = None
            match_end_index = -1
            k = i
            while k < n:
                char = text[k]
                if char not in node.children:
                    break
                node = node.children[char]
                if node.is_end:
                    longest_match = (i, k + 1, node.term, node.target)
                    match_end_index = k + 1
                k += 1
            if longest_match:
                matches.append(longest_match)
                i = match_end_index
            else:
                i += 1
        self.logger.debug(f"Found {len(matches)} non-overlapping glossary terms in text.")
        return matches

    def get_target_term(self, source_term: str) -> Optional[str]:
        """Gets the target term for a source term."""
        if not isinstance(source_term, str):
            return None
        return self.glossary.get(source_term)

    def create_placeholders(self, text: str) -> Tuple[str, Dict[str, Tuple[str, str]]]:
        """Replaces glossary terms with placeholders in the text."""
        terms = self.find_terms_in_text(text)
        placeholder_map = {}
        processed_text = text

        # Sort terms by start index in reverse to avoid invalidating indices during replacement
        terms.sort(key=lambda x: x[0], reverse=True)

        for start, end, term, target in terms:
            if term != target and target and target.strip():
                placeholder = f"__GLOSSARY_{len(placeholder_map)}__"
                processed_text = processed_text[:start] + placeholder + processed_text[end:]
                placeholder_map[placeholder] = (term, target)
                self.logger.debug(f"Replaced '{term}' at [{start}:{end}] with '{placeholder}'.")

        self.logger.debug(f"Created {len(placeholder_map)} placeholders for glossary terms.")
        return processed_text, placeholder_map

    def restore_placeholders(self, text: str, placeholder_map: Dict[str, Tuple[str, str]]) -> str:
        """Restores glossary terms (target values) from placeholders."""
        restored_text = text
        # Sort placeholders by length in reverse to handle potential nested or similar placeholders correctly
        sorted_placeholders = sorted(placeholder_map.items(), key=lambda item: len(item[0]), reverse=True)

        for placeholder, (_, target) in sorted_placeholders:
            restored_text = restored_text.replace(placeholder, target)

        self.logger.debug(f"Attempted to restore {len(placeholder_map)} placeholders.")
        return restored_text

    def _validate_glossary_restoration(self, original_text_with_placeholders: str, translated_text_with_placeholders: str, placeholder_map: Dict[str, Tuple[str, str]]) -> bool:
        """
        Validates if all placeholders inserted *before* translation are still present in the *translated* text before restoration.
        This checks if the model followed the instruction to preserve placeholders.
        It does NOT check the correctness of the terms *after* restoration.
        """
        if not placeholder_map:
            return True

        errors = []
        placeholders_preserved = True
        for placeholder in placeholder_map.keys():
            if placeholder not in translated_text_with_placeholders:
                errors.append(f"Placeholder '{placeholder}' was removed during translation.")
                placeholders_preserved = False

        if errors:
            self.logger.warning(f"Glossary placeholder preservation validation failed: {', '.join(errors)}")
            return False
        self.logger.debug("Glossary placeholder preservation validation passed.")
        return True