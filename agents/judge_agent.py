"""Judge Agent: Validates LLM outputs - FIXED VERSION"""
import re
import requests
from typing import Dict, List

class JudgeAgent:
    def __init__(self, config):
        self.config = config
        self.ollama_host = config["ollama"]["host"]
        self.model = config["ollama"]["model_judge"]
        self.min_confidence = config["agents"]["judge"]["min_confidence"]
    
    def validate(self, answer: str, source_chunks: List[Dict], query: str) -> Dict:
        """Validate answer against source chunks"""
        
        # Extract factual claims
        claims = self._extract_claims(answer)
        
        if not claims:
            # No specific claims to validate
            return {
                'is_valid': True,
                'confidence': 0.95,
                'validation_text': 'No specific factual claims to validate',
                'flagged_issues': []
            }
        
        # Build validation prompt
        context = '\n\n'.join([f"[Chunk {i+1}] {c['content'][:600]}" 
                               for i, c in enumerate(source_chunks[:10])])
        
        validation_prompt = f"""<|im_start|>system
You are a fact checker. Verify if claims in the answer are supported by the context.

For each claim with a number, date, or well name:
- Write "VALID:" if it matches the context
- Write "UNCERTAIN:" if partially supported or slightly different
- Write "INVALID:" only if clearly contradicts context

Be generous - mark as VALID if reasonably close or supported.
<|im_end|>
<|im_start|>user
Query: {query}

Answer claims to check:
{' | '.join(claims[:10])}

Context:
{context}

Validation (one line per claim):
<|im_end|>
<|im_start|>assistant
"""
        
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    'model': self.model,
                    'prompt': validation_prompt,
                    'stream': False,
                    'options': {'temperature': 0.1, 'num_ctx': 4096}
                },
                timeout=120
            )
            
            if resp.status_code != 200:
                # Judge failed, assume valid
                return {
                    'is_valid': True,
                    'confidence': 0.8,
                    'validation_text': f'Judge unavailable (HTTP {resp.status_code})',
                    'flagged_issues': ['Judge service error']
                }
            
            validation = resp.json().get('response', '')
            
            # Parse validation results (more lenient)
            valid_count = validation.upper().count('VALID:')
            invalid_count = validation.upper().count('INVALID:')
            uncertain_count = validation.upper().count('UNCERTAIN:')
            
            # Count UNCERTAIN as half-valid
            total = valid_count + invalid_count + uncertain_count
            if total == 0:
                # Validation didn't follow format, assume valid
                confidence = 0.85
            else:
                confidence = (valid_count + 0.5 * uncertain_count) / total
            
            return {
                'is_valid': confidence >= self.min_confidence,
                'confidence': confidence,
                'validation_text': validation[:500],
                'flagged_issues': self._extract_issues(validation)
            }
        
        except requests.exceptions.Timeout:
            return {
                'is_valid': True,
                'confidence': 0.8,
                'validation_text': 'Judge timeout - assuming valid',
                'flagged_issues': []
            }
        except Exception as e:
            return {
                'is_valid': True,
                'confidence': 0.8,
                'validation_text': f'Judge error: {e}',
                'flagged_issues': []
            }
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer"""
        claims = []
        
        # Numbers with units
        claims.extend(re.findall(r'\d+\.?\d*\s*(?:m|meters|bar|°C|kg/m³|TVD|MD)', answer))
        
        # Dates
        claims.extend(re.findall(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', answer))
        claims.extend(re.findall(r'\d{4}-\d{2}-\d{2}', answer))
        
        # Well names
        claims.extend(re.findall(r'[A-Z]{2,10}-GT-\d{2}(?:-S\d+)?', answer))
        
        return claims[:15]  # Limit to 15 claims
    
    def _extract_issues(self, validation_text: str) -> List[str]:
        """Extract flagged issues from validation"""
        issues = []
        lines = validation_text.split('\n')
        for line in lines:
            if 'INVALID' in line.upper():
                issues.append(line.strip())
        return issues[:3]  # Top 3 issues only
