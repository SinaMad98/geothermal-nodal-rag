"""Generic Ingestion Agent - Works with any well report format"""
import fitz
from pathlib import Path
import re

class IngestionAgent:
    def __init__(self, config):
        self.config = config
    
    def process(self, file_paths, log_capture=None):
        """Process PDFs with generic well name detection"""
        documents = []
        
        for fp in file_paths:
            try:
                if log_capture:
                    log_capture.add(f"📄 Processing: {Path(fp).name}", "INFO")
                
                pdf = fitz.open(fp)
                pages = []
                full_text = ""
                
                for i in range(len(pdf)):
                    text = pdf[i].get_text()
                    pages.append({
                        'page_number': i + 1,
                        'content': text
                    })
                    full_text += text
                
                # ENHANCEMENT 3: Generic well name extraction
                wells = self._extract_well_names(full_text)
                
                if log_capture:
                    log_capture.add(f"✓ {len(pages)} pages, Wells: {', '.join(wells) if wells else 'None'}", "INFO")
                
                documents.append({
                    'source_file': Path(fp).name,
                    'metadata': {
                        'source_file': Path(fp).name,
                        'well_names': ', '.join(wells)  # Convert to string for ChromaDB
                    },
                    'pages': pages
                })
                
                pdf.close()
                
            except Exception as e:
                if log_capture:
                    log_capture.add(f"❌ Failed: {Path(fp).name} - {str(e)[:100]}", "ERROR")
        
        return {'documents': documents}
    
    def _extract_well_names(self, text: str) -> list:
        """Extract well names using multiple patterns"""
        well_names = set()
        
        # Pattern 1: Standard (ADK-GT-01, NLW-GT-03-S1, HAG-GT-01-02)
        pattern1 = r'\b([A-Z]{2,4}-GT-\d{2}(?:-S\d+)?(?:-\d{2})?)\b'
        wells1 = re.findall(pattern1, text)
        well_names.update(wells1)
        
        # Pattern 2: Naaldwijk style (NAALDWIJK-GT-02-S1)
        pattern2 = r'\b([A-Z]{4,}-GT-\d{2}(?:-S\d+)?)\b'
        wells2 = re.findall(pattern2, text)
        well_names.update(wells2)
        
        # Pattern 3: With spaces (HAG GT 01)
        pattern3 = r'\b([A-Z]{2,4})\s+GT\s+(\d{2})\b'
        for match in re.finditer(pattern3, text):
            well_name = f"{match.group(1)}-GT-{match.group(2)}"
            well_names.add(well_name)
        
        return sorted(list(well_names))
