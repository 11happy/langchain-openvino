from queue import Queue
import re

import numpy as np
import openvino as ov
import openvino_genai as ov_genai

core = ov.Core()

class OpenVINOStreamer(ov_genai.StreamerBase):
    """OpenVINO streamer for token-by-token generation."""
    
    def __init__(self, tokenizer, detok_path: str):
        """Initialize streamer with tokenizer and detokenizer.
        
        Args:
            tokenizer: The tokenizer to use
            detok_path: Path to the OpenVINO detokenizer model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.detok = core.compile_model(detok_path, "CPU")
        self.q = Queue()
        self.stop = object()
    
    def __iter__(self):
        """Return iterator."""
        return self
    
    def __next__(self):
        """Get next token."""
        val = self.q.get()
        if val == self.stop:
            raise StopIteration()
        return val
    
    def put(self, token_id):
        """Process token and add to queue.
        
        Args:
            token_id: Token ID to process
        """
        out = self.detok(np.array([[token_id]], dtype=int))
        txt = str(out["string_output"][0])
        txt = txt.lstrip("!")
        txt = re.sub(r"<.*?>", "", txt)
        self.q.put(txt)
    
    def end(self):
        """Signal end of generation."""
        self.q.put(self.stop)