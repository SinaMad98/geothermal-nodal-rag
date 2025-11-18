"""Enhanced Parameter Extraction: Advanced Trajectory Detection"""
import re
from typing import List, Dict
import requests

class ParameterExtractionAgent:
    def __init__(self, config):
        self.config = config
        self.ollama_host = config['ollama']['host']
        self.model = config['ollama']['model_extraction']
    
    def extract(self, chunks: List[Dict], log_capture=None) -> Dict:
        """Extract trajectory with advanced table detection"""
        
        if log_capture:
            log_capture.add("🔧 Enhanced trajectory extraction starting...", "INFO")
        
        # Step 1: Find trajectory tables using multiple patterns
        trajectory_chunks = self._detect_trajectory_tables(chunks)
        
        if not trajectory_chunks:
            if log_capture:
                log_capture.add("⚠️  No trajectory tables detected", "WARN")
            return {'trajectory': [], 'confidence': 0.0}
        
        if log_capture:
            log_capture.add(f"✓ Found {len(trajectory_chunks)} potential trajectory chunks", "INFO")
        
        # Step 2: Extract MD/TVD/ID data using regex
        trajectory_points = []
        for chunk in trajectory_chunks[:10]:  # Process top 10 candidates
            points = self._extract_trajectory_points(chunk['content'])
            trajectory_points.extend(points)
        
        if not trajectory_points:
            # Step 3: Fallback - use LLM extraction
            if log_capture:
                log_capture.add("🤖 Using LLM for trajectory extraction", "INFO")
            trajectory_points = self._llm_extract_trajectory(trajectory_chunks[:5])
        
        # Step 4: Clean and validate
        trajectory_points = self._clean_trajectory(trajectory_points)
        
        if log_capture:
            log_capture.add(f"✅ Extracted {len(trajectory_points)} trajectory points", "SUCCESS")
        
        return {
            'trajectory': trajectory_points,
            'confidence': min(0.9, len(trajectory_points) / 50),  # Confidence based on point count
            'source_chunks': len(trajectory_chunks)
        }
    
    def _detect_trajectory_tables(self, chunks: List[Dict]) -> List[Dict]:
        """Detect chunks containing trajectory data"""
        trajectory_patterns = [
            r'(?i)(measured\s+depth|MD|depth\s+\(m\))',
            r'(?i)(true\s+vertical\s+depth|TVD)',
            r'(?i)(inclination|azimuth|angle)',
            r'MD\s*TVD\s*(?:Incl|Inc|Angle)',
            r'\d{3,4}\.\d+\s+\d{3,4}\.\d+\s+\d+\.\d+'  # Pattern: 1000.5 950.2 15.3
        ]
        
        scored_chunks = []
        for chunk in chunks:
            content = chunk['content']
            score = 0
            
            # Score based on pattern matches
            for pattern in trajectory_patterns:
                if re.search(pattern, content):
                    score += 1
            
            # Bonus: Has table structure
            if '|' in content or '\t' in content:
                score += 2
            
            # Bonus: Multiple numeric lines
            numeric_lines = len(re.findall(r'\d{3,4}\.?\d*\s+\d{3,4}\.?\d*', content))
            score += min(numeric_lines, 5)
            
            if score >= 3:
                scored_chunks.append({'chunk': chunk, 'score': score})
        
        # Sort by score and return top candidates
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return [item['chunk'] for item in scored_chunks[:15]]
    
    def _extract_trajectory_points(self, text: str) -> List[Dict]:
        """Extract MD/TVD/ID from text using regex"""
        points = []
        
        # Pattern 1: Table format with | separators
        table_pattern = r'\|\s*(\d{1,4}\.?\d*)\s*\|\s*(\d{1,4}\.?\d*)\s*\|\s*(\d+\.?\d*)'
        matches = re.findall(table_pattern, text)
        for md, tvd, inc_or_id in matches:
            try:
                points.append({
                    'MD': float(md),
                    'TVD': float(tvd),
                    'ID': float(inc_or_id) if float(inc_or_id) < 90 else 0.0  # Filter inclination
                })
            except ValueError:
                continue
        
        # Pattern 2: Space-separated (more common)
        space_pattern = r'(?:^|\n)\s*(\d{3,4}\.?\d*)\s+(\d{3,4}\.?\d*)\s+(\d+\.?\d*)'
        matches = re.findall(space_pattern, text, re.MULTILINE)
        for md, tvd, inc_or_id in matches:
            try:
                md_val, tvd_val = float(md), float(tvd)
                # Validate: TVD should be <= MD
                if tvd_val <= md_val and md_val < 5000:
                    points.append({
                        'MD': md_val,
                        'TVD': tvd_val,
                        'ID': float(inc_or_id) if float(inc_or_id) < 90 else 0.0
                    })
            except ValueError:
                continue
        
        return points
    
    def _llm_extract_trajectory(self, chunks: List[Dict]) -> List[Dict]:
        """Fallback: Use LLM to extract trajectory"""
        context = '\n\n'.join([c['content'][:800] for c in chunks[:3]])
        
        prompt = f"""Extract trajectory data from the following well report.

Find all MD (Measured Depth), TVD (True Vertical Depth), and ID (Inner Diameter) values.

Context:
{context}

Return ONLY comma-separated values in format: MD,TVD,ID
Example:
1000.5,950.2,0.216
2000.0,1850.3,0.216
"""
        
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={'model': self.model, 'prompt': prompt, 'stream': False,
                      'options': {'temperature': 0.1, 'num_ctx': 4096}},
                timeout=120
            )
            
            result = resp.json().get('response', '')
            points = []
            
            for line in result.split('\n'):
                if ',' in line:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            points.append({
                                'MD': float(parts[0]),
                                'TVD': float(parts[1]),
                                'ID': float(parts[2]) if len(parts) > 2 else 0.216
                            })
                    except (ValueError, IndexError):
                        continue
            
            return points
        
        except Exception as e:
            return []
    
    def _clean_trajectory(self, points: List[Dict]) -> List[Dict]:
        """Clean and deduplicate trajectory points"""
        # Remove duplicates
        seen = set()
        cleaned = []
        for p in points:
            key = (round(p['MD'], 1), round(p['TVD'], 1))
            if key not in seen:
                seen.add(key)
                cleaned.append(p)
        
        # Sort by MD
        cleaned.sort(key=lambda x: x['MD'])
        
        # Validate sequence
        final = []
        prev_md = 0
        for p in cleaned:
            if p['MD'] > prev_md and p['TVD'] <= p['MD']:
                final.append(p)
                prev_md = p['MD']
        
        return final[:100]  # Limit to 100 points
