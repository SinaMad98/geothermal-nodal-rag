"""Enhanced RAG Retrieval - FIXED metadata filtering"""
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
import time

class RAGRetrievalAgent:
    def __init__(self, config):
        self.config = config
        self.client = chromadb.PersistentClient(path=config['vector_store']['path'])
        self.collections = {}
        self.bm25_indices = {}
        self.documents_cache = {}
    
    def index_documents(self, chunks, strategy='factual_qa'):
        collection_name = self.config['embedding_strategies'][strategy]['collection']
        
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"strategy": strategy}
            )
        
        collection = self.collections[collection_name]
        
        documents = [c['content'] for c in chunks]
        metadatas = [c['metadata'] for c in chunks]
        ids = [f"{strategy}_{i}_{int(time.time())}" for i in range(len(chunks))]
        
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        # BM25 index
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25_indices[collection_name] = BM25Okapi(tokenized_docs)
        self.documents_cache[collection_name] = chunks
        
        print(f"  ✅ Indexed {len(chunks)} chunks into {collection_name} (Hybrid)")
    
    def retrieve(self, query, mode='qa', well_name=None):
        """Hybrid retrieval - FIXED: removed $contains filter"""
        strategy_map = {'qa': 'factual_qa', 'extract': 'technical_extraction', 'summary': 'summary'}
        strategy = strategy_map.get(mode, 'factual_qa')
        collection_name = self.config['embedding_strategies'][strategy]['collection']
        top_k_key = f"top_k_{mode if mode != 'qa' else 'factual'}"
        top_k = self.config['agents']['rag_retrieval'][top_k_key]
        
        collection = self.collections.get(collection_name)
        if not collection:
            return {'chunks': []}
        
        # Semantic search (NO metadata filtering - ChromaDB limitation)
        semantic_results = collection.query(
            query_texts=[query],
            n_results=top_k * 2
        )
        
        # BM25 keyword search
        bm25 = self.bm25_indices.get(collection_name)
        keyword_scores = []
        if bm25:
            tokenized_query = query.lower().split()
            keyword_scores = bm25.get_scores(tokenized_query)
        
        # Hybrid fusion
        semantic_weight = self.config['agents']['rag_retrieval']['hybrid_weight_semantic']
        keyword_weight = self.config['agents']['rag_retrieval']['hybrid_weight_keyword']
        
        chunks = []
        for i, (doc, meta, distance) in enumerate(zip(
            semantic_results['documents'][0],
            semantic_results['metadatas'][0],
            semantic_results['distances'][0]
        )):
            # FIXED: Filter by well name AFTER retrieval (in Python)
            if well_name:
                well_names_str = meta.get('well_names', '')
                if well_name.upper() not in well_names_str.upper():
                    continue  # Skip chunks not matching target well
            
            semantic_score = 1 / (1 + distance)
            keyword_score = keyword_scores[i] if i < len(keyword_scores) else 0
            
            combined_score = (semantic_weight * semantic_score + keyword_weight * keyword_score)
            
            chunks.append({
                'content': doc,
                'metadata': meta,
                'score': combined_score,
                'citation': f"{meta.get('source_file', 'Unknown')} p.{meta.get('page_number', '?')}"
            })
        
        chunks.sort(key=lambda x: x['score'], reverse=True)
        return {'chunks': chunks[:top_k]}
