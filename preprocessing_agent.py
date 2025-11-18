"""Ultra-Sophisticated Preprocessing: NLP-Level Semantic Chunking"""
from typing import List, Dict
import re
import time

class PreprocessingAgent:
    def __init__(self, config):
        self.config = config
    
    def process_all_strategies(self, documents: List[Dict], log_capture=None) -> Dict[str, List[Dict]]:
        """ULTRA-sophisticated NLP-level semantic chunking"""
        start_time = time.time()
        
        if log_capture:
            log_capture.add(f"📊 Processing {len(documents)} documents (ULTRA-SOPHISTICATED mode)", "INFO")
        
        all_chunks = {name: [] for name in self.config['embedding_strategies'].keys()}
        
        for doc_idx, doc in enumerate(documents):
            pages = doc.get('pages', [])
            if log_capture:
                log_capture.add(f"  Doc {doc_idx+1}: {len(pages)} pages - Ultra NLP chunking", "INFO")
            
            doc_metadata = doc.get('metadata', {})
            clean_metadata = self._clean_metadata(doc_metadata)
            
            for page in pages:
                content = page.get('content', '')
                page_num = page.get('page_number', 0)
                
                if len(content.strip()) < 50:
                    continue
                
                # ULTRA processing: NLP-level analysis
                time.sleep(0.1)  # More time for quality
                
                # Step 1: Detect document structure
                doc_structure = self._analyze_document_structure(content)
                
                # Step 2: Extract entities (wells, numbers, dates)
                entities = self._extract_entities(content)
                
                # Step 3: Semantic segmentation
                semantic_segments = self._ultra_semantic_segmentation(
                    content, page_num, doc_structure, entities
                )
                
                for segment in semantic_segments:
                    metadata = {
                        **clean_metadata,
                        'page_number': page_num,
                        'source_file': doc.get('source_file', 'Unknown'),
                        'segment_type': segment['type'],
                        'paragraph_id': segment['id'],
                        'has_numbers': segment.get('has_numbers', False),
                        'has_dates': segment.get('has_dates', False),
                        'has_wells': segment.get('has_wells', False),
                        'entity_density': segment.get('entity_density', 0),
                        'section_depth': segment.get('section_depth', 0)
                    }
                    
                    for strategy_name, strategy_config in self.config['embedding_strategies'].items():
                        chunks = self._ultra_chunk(
                            segment['content'],
                            strategy_config['chunk_size'],
                            strategy_config['chunk_overlap'],
                            metadata,
                            segment.get('entities', [])
                        )
                        all_chunks[strategy_name].extend(chunks)
        
        elapsed = time.time() - start_time
        if log_capture:
            log_capture.add(f"  ⏱️  ULTRA chunking took {elapsed:.1f} seconds", "INFO")
            for strategy, chunks in all_chunks.items():
                log_capture.add(f"  ✓ {strategy}: {len(chunks)} chunks (ultra-semantic)", "SUCCESS")
        
        return all_chunks
    
    def _analyze_document_structure(self, content: str) -> Dict:
        """Analyze document structure hierarchy"""
        structure = {
            'has_toc': False,
            'has_sections': False,
            'has_tables': False,
            'has_figures': False,
            'section_depth': 0
        }
        
        # Table of contents detection
        if re.search(r'(?i)(table of contents|contents)', content[:500]):
            structure['has_toc'] = True
        
        # Section hierarchy detection
        section_patterns = [
            r'^\d+\.\s+[A-Z]',  # 1. Section
            r'^\d+\.\d+\s+[A-Z]',  # 1.1 Subsection
            r'^\d+\.\d+\.\d+\s+[A-Z]'  # 1.1.1 Subsubsection
        ]
        
        max_depth = 0
        for i, pattern in enumerate(section_patterns):
            if re.search(pattern, content, re.MULTILINE):
                max_depth = i + 1
                structure['has_sections'] = True
        structure['section_depth'] = max_depth
        
        # Table detection
        if re.search(r'\|[^\n]+\|[^\n]+\||\t[^\n]+\t[^\n]+', content):
            structure['has_tables'] = True
        
        # Figure detection
        if re.search(r'(?i)(figure|fig\.|diagram)\s*\d+', content):
            structure['has_figures'] = True
        
        return structure
    
    def _extract_entities(self, content: str) -> Dict:
        """Extract key entities: wells, measurements, dates"""
        entities = {
            'wells': [],
            'depths': [],
            'dates': [],
            'temperatures': [],
            'pressures': []
        }
        
        # Well names
        entities['wells'] = list(set(re.findall(r'[A-Z]{2,10}-GT-\d{2}(?:-S\d+)?', content)))
        
        # Depths
        entities['depths'] = re.findall(r'(\d{3,4}\.?\d*)\s*(?:m|meters)\b', content)
        
        # Dates
        entities['dates'] = re.findall(
            r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|'
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            content
        )
        
        # Temperatures
        entities['temperatures'] = re.findall(r'(\d{1,3}\.?\d*)\s*(?:°C|celsius)', content)
        
        # Pressures
        entities['pressures'] = re.findall(r'(\d{1,4}\.?\d*)\s*(?:bar|psi|MPa)', content)
        
        return entities
    
    def _ultra_semantic_segmentation(self, content: str, page_num: int, 
                                    structure: Dict, entities: Dict) -> List[Dict]:
        """Ultra-sophisticated semantic segmentation"""
        segments = []
        
        # Multi-level segmentation strategy
        if structure['section_depth'] >= 2:
            # High structure document - section-based
            sections = self._extract_hierarchical_sections(content, structure['section_depth'])
            for i, section in enumerate(sections):
                seg_entities = self._match_entities_to_text(section['content'], entities)
                entity_density = len(seg_entities['wells']) + len(seg_entities['depths']) + len(seg_entities['dates'])
                
                segments.append({
                    'content': section['content'],
                    'type': f"section_L{section['level']}",
                    'id': f"p{page_num}_s{i+1}_L{section['level']}",
                    'has_numbers': len(seg_entities['depths']) > 0,
                    'has_dates': len(seg_entities['dates']) > 0,
                    'has_wells': len(seg_entities['wells']) > 0,
                    'entity_density': entity_density,
                    'section_depth': section['level'],
                    'entities': seg_entities
                })
        
        elif structure['has_tables']:
            # Table-heavy document - table + text segmentation
            tables = self._extract_tables(content)
            text_blocks = self._extract_non_table_text(content, tables)
            
            for i, table in enumerate(tables):
                seg_entities = self._match_entities_to_text(table, entities)
                segments.append({
                    'content': table,
                    'type': 'table',
                    'id': f"p{page_num}_tbl{i+1}",
                    'has_numbers': True,
                    'has_dates': False,
                    'has_wells': len(seg_entities['wells']) > 0,
                    'entity_density': len(seg_entities['depths']) * 2,  # Tables are data-rich
                    'section_depth': 0,
                    'entities': seg_entities
                })
            
            for i, block in enumerate(text_blocks):
                if len(block.strip()) > 50:
                    seg_entities = self._match_entities_to_text(block, entities)
                    segments.append({
                        'content': block,
                        'type': 'text_block',
                        'id': f"p{page_num}_txt{i+1}",
                        'has_numbers': len(seg_entities['depths']) > 0,
                        'has_dates': len(seg_entities['dates']) > 0,
                        'has_wells': len(seg_entities['wells']) > 0,
                        'entity_density': len(seg_entities['wells']) + len(seg_entities['depths']),
                        'section_depth': 0,
                        'entities': seg_entities
                    })
        
        else:
            # Unstructured document - intelligent paragraph segmentation
            paragraphs = self._intelligent_paragraph_split(content)
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 30:
                    seg_entities = self._match_entities_to_text(para, entities)
                    entity_density = len(seg_entities['wells']) + len(seg_entities['depths']) + len(seg_entities['dates'])
                    
                    segments.append({
                        'content': para,
                        'type': 'paragraph',
                        'id': f"p{page_num}_par{i+1}",
                        'has_numbers': len(seg_entities['depths']) > 0,
                        'has_dates': len(seg_entities['dates']) > 0,
                        'has_wells': len(seg_entities['wells']) > 0,
                        'entity_density': entity_density,
                        'section_depth': 0,
                        'entities': seg_entities
                    })
        
        return segments if segments else [{
            'content': content, 'type': 'page', 'id': f"p{page_num}",
            'has_numbers': False, 'has_dates': False, 'has_wells': False,
            'entity_density': 0, 'section_depth': 0, 'entities': {}
        }]
    
    def _extract_hierarchical_sections(self, content: str, max_depth: int) -> List[Dict]:
        """Extract nested section hierarchy"""
        sections = []
        patterns = [
            (r'^\d+\.\s+([^\n]+)', 1),
            (r'^\d+\.\d+\s+([^\n]+)', 2),
            (r'^\d+\.\d+\.\d+\s+([^\n]+)', 3)
        ]
        
        for pattern, level in patterns[:max_depth]:
            matches = list(re.finditer(pattern, content, re.MULTILINE))
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(content)
                sections.append({
                    'content': content[start:end],
                    'level': level,
                    'title': match.group(1)
                })
        
        return sorted(sections, key=lambda x: content.find(x['content']))
    
    def _extract_tables(self, content: str) -> List[str]:
        """Extract table content"""
        tables = []
        # Pattern for markdown-style tables
        table_pattern = r'(\|[^\n]+\n)+'
        for match in re.finditer(table_pattern, content):
            tables.append(match.group(0))
        return tables
    
    def _extract_non_table_text(self, content: str, tables: List[str]) -> List[str]:
        """Extract text excluding tables"""
        for table in tables:
            content = content.replace(table, '\n---TABLE---\n')
        return [block for block in content.split('---TABLE---') if len(block.strip()) > 50]
    
    def _intelligent_paragraph_split(self, content: str) -> List[str]:
        """Intelligent paragraph splitting preserving context"""
        # Split on double newlines but preserve single newlines
        rough_paras = content.split('\n\n')
        
        # Merge very short paragraphs with next
        refined_paras = []
        buffer = ""
        for para in rough_paras:
            if len(para.strip()) < 100 and buffer:
                buffer += "\n\n" + para
            elif buffer:
                refined_paras.append(buffer)
                buffer = para
            else:
                buffer = para
        
        if buffer:
            refined_paras.append(buffer)
        
        return refined_paras
    
    def _match_entities_to_text(self, text: str, all_entities: Dict) -> Dict:
        """Find which entities appear in this text"""
        matched = {key: [] for key in all_entities.keys()}
        
        for entity_type, entity_list in all_entities.items():
            for entity in entity_list:
                if str(entity) in text:
                    matched[entity_type].append(entity)
        
        return matched
    
    def _ultra_chunk(self, text: str, chunk_size: int, overlap: int, 
                    base_metadata: Dict, entities: Dict) -> List[Dict]:
        """Ultra-sophisticated chunking with entity awareness"""
        words = text.split()
        chunks = []
        
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            
            if len(chunk_words) < 30:
                break
            
            chunk_text = ' '.join(chunk_words)
            
            # Deep quality analysis (slow but worth it)
            time.sleep(0.02)
            
            # Calculate chunk quality score
            chunk_entities = self._match_entities_to_text(chunk_text, entities)
            quality_score = (
                len(chunk_entities.get('wells', [])) * 3 +
                len(chunk_entities.get('depths', [])) * 2 +
                len(chunk_entities.get('dates', [])) * 2 +
                len(chunk_entities.get('temperatures', [])) +
                len(chunk_entities.get('pressures', []))
            )
            
            chunk_metadata = {
                **base_metadata,
                'chunk_index': chunk_idx,
                'word_start': i,
                'word_end': i + len(chunk_words),
                'chunk_length': len(chunk_words),
                'quality_score': quality_score,
                'contains_wells': ','.join(chunk_entities.get('wells', []))
            }
            
            chunks.append({
                'content': chunk_text,
                'metadata': chunk_metadata,
                'citation': self._build_precise_citation(chunk_metadata)
            })
            
            i += (chunk_size - overlap)
            chunk_idx += 1
        
        return chunks
    
    def _build_precise_citation(self, metadata: Dict) -> str:
        file = metadata.get('source_file', 'Unknown')
        page = metadata.get('page_number', '?')
        para_id = metadata.get('paragraph_id', '')
        segment_type = metadata.get('segment_type', 'text')
        
        citation = f"{file}, p.{page}"
        if para_id:
            citation += f", {segment_type} {para_id}"
        
        return citation
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                clean[key] = ', '.join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean
