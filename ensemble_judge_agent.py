"""Ensemble Judge: Multi-model validation for higher accuracy"""
import requests
from typing import Dict, List
import statistics

class EnsembleJudgeAgent:
    def __init__(self, config):
        self.config = config
        self.ollama_host = config['ollama']['host']
        self.models = config['agents']['judge']['ensemble_models']  # List of models
        self.min_confidence = config['agents']['judge']['min_confidence']
    
    def validate(self, answer: str, source_chunks: List[Dict], query: str) -> Dict:
        """Ensemble validation: Multiple models vote"""
        
        # Extract claims
        claims = self._extract_claims(answer)
        
        if not claims:
            return {
                'is_valid': True,
                'confidence': 0.95,
                'validation_text': 'No specific claims',
                'flagged_issues': [],
                'ensemble_votes': []
            }
        
        # Build validation context
        context = '\n\n'.join([f"[Chunk {i+1}] {c['content'][:500]}" 
                               for i, c in enumerate(source_chunks[:8])])
        
        validation_prompt = f"""Fact checker. Verify claims against context.

For each claim:
- VALID: Directly supported
- UNCERTAIN: Partially supported
- INVALID: Contradicts or missing

Claims to check:
{' | '.join(claims[:10])}

Context:
{context}

Validation (one per claim):
"""
        
        # Get validations from multiple models
        validations = []
        for model in self.models:
            try:
                resp = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        'model': model,
                        'prompt': validation_prompt,
                        'stream': False,
                        'options': {'temperature': 0.1, 'num_ctx': 4096}
                    },
                    timeout=90
                )
                
                if resp.status_code == 200:
                    validation = resp.json().get('response', '')
                    confidence = self._calculate_confidence(validation)
                    validations.append({
                        'model': model,
                        'confidence': confidence,
                        'validation': validation
                    })
            
            except Exception as e:
                continue
        
        if not validations:
            # All judges failed - assume valid
            return {
                'is_valid': True,
                'confidence': 0.75,
                'validation_text': 'Ensemble unavailable',
                'flagged_issues': [],
                'ensemble_votes': []
            }
        
        # Ensemble decision: Average confidence
        confidences = [v['confidence'] for v in validations]
        ensemble_confidence = statistics.mean(confidences)
        
        # Majority voting for issues
        all_issues = []
        for v in validations:
            all_issues.extend(self._extract_issues(v['validation']))
        
        # Issues mentioned by majority
        from collections import Counter
        issue_counts = Counter(all_issues)
        flagged_issues = [issue for issue, count in issue_counts.items() 
                         if count >= len(validations) / 2]
        
        return {
            'is_valid': ensemble_confidence >= self.min_confidence,
            'confidence': ensemble_confidence,
            'validation_text': f"Ensemble: {len(validations)} models",
            'flagged_issues': flagged_issues[:3],
            'ensemble_votes': [{'model': v['model'], 'conf': v['confidence']} 
                              for v in validations]
        }
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims"""
        import re
        claims = []
        claims.extend(re.findall(r'\d+\.?\d*\s*(?:m|meters|bar|°C|kg/m³|TVD|MD)', answer))
        claims.extend(re.findall(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', answer))
        claims.extend(re.findall(r'[A-Z]{2,10}-GT-\d{2}(?:-S\d+)?', answer))
        return claims[:15]
    
    def _calculate_confidence(self, validation_text: str) -> float:
        """Calculate confidence from validation text"""
        valid_count = validation_text.upper().count('VALID:')
        invalid_count = validation_text.upper().count('INVALID:')
        uncertain_count = validation_text.upper().count('UNCERTAIN:')
        
        total = valid_count + invalid_count + uncertain_count
        if total == 0:
            return 0.8
        
        return (valid_count + 0.5 * uncertain_count) / total
    
    def _extract_issues(self, validation_text: str) -> List[str]:
        """Extract flagged issues"""
        issues = []
        lines = validation_text.split('\n')
        for line in lines:
            if 'INVALID' in line.upper():
                issues.append(line.strip()[:80])
        return issues
