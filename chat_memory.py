"""Chat Memory: Multi-turn conversation support"""
from typing import List, Dict
from datetime import datetime

class ChatMemory:
    def __init__(self, config):
        self.buffer_size = config['agents']['memory']['buffer_size']
        self.buffer = []  # Recent messages
        self.well_context = {}  # Per-well facts
        self.enable_well_context = config['agents']['memory']['enable_well_context']
    
    def add_turn(self, query: str, answer: str, metadata: Dict = None):
        """Add query-answer pair to memory"""
        turn = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'answer': answer[:500],  # Truncate long answers
            'metadata': metadata or {}
        }
        
        self.buffer.append(turn)
        
        # Keep only last N turns
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        # Extract and store well-specific facts
        if self.enable_well_context and metadata:
            well_name = metadata.get('well_name')
            if well_name:
                if well_name not in self.well_context:
                    self.well_context[well_name] = {}
                
                # Store facts about this well
                if 'depth' in answer.lower():
                    import re
                    depth_match = re.search(r'(\d{3,4})\s*m', answer)
                    if depth_match:
                        self.well_context[well_name]['depth'] = depth_match.group(1)
    
    def get_context(self, query: str) -> str:
        """Get relevant context from memory"""
        if not self.buffer:
            return ""
        
        # Build context from recent turns
        context_parts = []
        for turn in self.buffer[-3:]:  # Last 3 turns
            context_parts.append(f"Previous Q: {turn['query'][:100]}\nA: {turn['answer'][:200]}")
        
        # Add well-specific facts if query mentions a well
        import re
        well_matches = re.findall(r'([A-Z]{2,10}-GT-\d{2}(?:-S\d+)?)', query)
        for well in well_matches:
            if well in self.well_context:
                facts = self.well_context[well]
                context_parts.append(f"\nKnown facts about {well}: {facts}")
        
        return "\n\n".join(context_parts)
    
    def clear(self):
        """Clear conversation memory"""
        self.buffer = []
        self.well_context = {}
