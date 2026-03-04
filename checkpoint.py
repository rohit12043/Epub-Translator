import json
import logging
from pathlib import Path
from typing import Dict, Optional


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "checkpoints", checkpoint_file: str = "translation_checkpoint.json"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / checkpoint_file
        self.checkpoint_data = {}
        self.logger = logging.getLogger(__name__)
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
            try:
                if self.checkpoint_file.exists():
                    with self.checkpoint_file.open('r', encoding='utf-8') as f:
                        self.checkpoint_data = json.load(f)
                    self.logger.info(f"Checkpoint loaded from {self.checkpoint_file}")
                else:
                    self.logger.info("No checkpoint file found, starting from scratch")
                    self.checkpoint_data = {}
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.checkpoint_data = {}

    def save_checkpoint(self, checkpoint_key: str, item_id: str, chunk_key: str = None, 
                        translated_line: str = None, line_index: int = None, 
                        is_dialogue: bool = False, completed: bool = False, 
                        tokens_used: int = 0, requests_made: int = 0, 
                        original_chunk_hash: str = None) -> None:
        try:
            if checkpoint_key not in self.checkpoint_data:
                self.checkpoint_data[checkpoint_key] = {}
            if item_id not in self.checkpoint_data[checkpoint_key]:
                self.checkpoint_data[checkpoint_key][item_id] = {'chunks': {}, 'completed': False}
                
            if chunk_key:
                if chunk_key not in self.checkpoint_data[checkpoint_key][item_id]['chunks']:
                    self.checkpoint_data[checkpoint_key][item_id]['chunks'][chunk_key] = {
                        'lines': {}, 
                        'original_hash': original_chunk_hash,
                        'completed_chunk': False 
                    }
                if translated_line is not None and line_index is not None:
                    line_data = {
                        'text': translated_line,
                        'is_dialogue': is_dialogue,
                        'tokens_used': tokens_used,
                        'requests_made': requests_made
                    }
                    self.checkpoint_data[checkpoint_key][item_id]['chunks'][chunk_key]['lines'][f"line{line_index}"] = line_data
                    
                if completed:
                    self.checkpoint_data[checkpoint_key][item_id]['chunks'][chunk_key]['completed_chunk'] = True
                    
            self.checkpoint_data[checkpoint_key][item_id]['completed'] = completed
            
            temp_file = self.checkpoint_file.with_suffix('.json.tmp')
            with temp_file.open('w', encoding='utf-8') as f:
                json.dump(self.checkpoint_data, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.checkpoint_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            
    def set_completed(self, checkpoint_key: str, item_id: str) -> None:
        self.save_checkpoint(checkpoint_key, item_id, completed=True)

    def get_chunk_data(self, checkpoint_key: str, item_id: str, chunk_key: str) -> Optional[Dict]:
        return self.checkpoint_data.get(checkpoint_key, {}).get(item_id, {}).get('chunks', {}).get(chunk_key)

    def get_item_completion_status(self, checkpoint_key: str, item_id: str) -> bool:
        return self.checkpoint_data.get(checkpoint_key, {}).get(item_id, {}).get('completed', False)
