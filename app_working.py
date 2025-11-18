"""Geothermal RAG v5.0 - Ensemble Judge + Enhanced Trajectory + Better Summaries"""
import gradio as gr
import yaml
import logging
from pathlib import Path
from datetime import datetime
import re
import requests
import subprocess

# Load v5.0 config
config = yaml.safe_load(open('config/config_v5.yaml'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

class LogCapture:
    def __init__(self):
        self.logs = []
    
    def add(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        logging.info(message)
        return "\n".join(self.logs[-50:])

log_capture = LogCapture()

class DocumentStore:
    def __init__(self):
        from agents.chat_memory import ChatMemory
        from agents.ensemble_judge_agent import EnsembleJudgeAgent
        from agents.fact_checking_agent import FactCheckingAgent
        
        self.rag_agent = None
        self.documents = []
        self.indexed_files = set()
        self.wells = set()
        self.log_capture = log_capture
        self.memory = ChatMemory(config)
        
        # Use Ensemble Judge if enabled, otherwise fallback
        if config['agents']['judge'].get('use_ensemble', False):
            self.judge = EnsembleJudgeAgent(config)
            self.log_capture.add("🎯 Ensemble Judge enabled", "INFO")
        else:
            from agents.judge_agent import JudgeAgent
            self.judge = JudgeAgent(config)
            self.log_capture.add("⚖️  Single Judge mode", "INFO")
        
        self.fact_checker = FactCheckingAgent(config)
        self.last_trajectory = None

    def index_documents(self, files):
        from agents.ingestion_agent import IngestionAgent
        from agents.preprocessing_agent import PreprocessingAgent
        from agents.rag_retrieval_agent import RAGRetrievalAgent

        if not files:
            return "No files selected.", "\n".join(self.log_capture.logs[-50:])
        
        self.log_capture.add(f"📤 Starting upload of {len(files)} files", "INFO")
        file_paths = [f.name for f in files]
        new_files = [f for f in file_paths if Path(f).name not in self.indexed_files]
        
        if not new_files:
            msg = f"Already indexed. Wells: {', '.join(self.wells)}"
            self.log_capture.add("⚠️  All files already indexed", "WARN")
            return msg, "\n".join(self.log_capture.logs[-50:])

        try:
            self.log_capture.add(f"📄 Processing {len(new_files)} new files", "INFO")
            ing = IngestionAgent(config)
            docs = ing.process(new_files, self.log_capture)
            
            self.log_capture.add(f"✓ Extracted {sum(len(d['pages']) for d in docs['documents'])} pages", "SUCCESS")
            
            # Ultra-sophisticated preprocessing
            prep = PreprocessingAgent(config)
            all_chunks = prep.process_all_strategies(docs['documents'], self.log_capture)
            
            if self.rag_agent is None:
                self.rag_agent = RAGRetrievalAgent(config)
                self.log_capture.add("🔧 RAGRetrievalAgent: Hybrid mode initialized", "INFO")
            
            total = 0
            for strategy, chunks in all_chunks.items():
                self.log_capture.add(f"💾 Indexing {len(chunks)} chunks for {strategy}", "INFO")
                self.rag_agent.index_documents(chunks, strategy=strategy)
                total += len(chunks)
            
            self.documents.extend(docs['documents'])
            self.indexed_files.update([Path(f).name for f in new_files])
            
            for d in docs['documents']:
                wells = d['metadata'].get('well_names', '')
                if isinstance(wells, str):
                    self.wells.update(w.strip() for w in wells.split(',') if w.strip())
            
            result = f"✅ Indexed {len(new_files)} files ({total} chunks)\n\n📊 Wells: {', '.join(sorted(self.wells))}"
            self.log_capture.add(f"✅ Indexing complete: {total} chunks.", "SUCCESS")
            return result, "\n".join(self.log_capture.logs[-50:])

        except Exception as e:
            import traceback
            err = f"Error: {e}\n{traceback.format_exc()[:400]}"
            self.log_capture.add(f"❌ ERROR: {str(e)[:200]}", "ERROR")
            return err, "\n".join(self.log_capture.logs[-50:])

    def query(self, user_query):
        if not self.documents:
            return "Please upload documents first.", "\n".join(self.log_capture.logs[-50:]), gr.update(visible=False)
        
        try:
            self.log_capture.add(f"🔍 Query: {user_query[:100]}", "INFO")
            
            # Get conversation context
            memory_context = self.memory.get_context(user_query)
            if memory_context:
                self.log_capture.add("💭 Using conversation history", "INFO")
            
            # Determine mode
            q = user_query.lower()
            mode = 'extract' if any(w in q for w in ['extract', 'trajectory']) else \
                   'summary' if any(w in q for w in ['summary', 'summarize']) else 'qa'
            
            self.log_capture.add(f"🎯 Mode: {mode.upper()}", "INFO")
            
            # Well-specific filtering
            target_well = next((w for w in self.wells if w.upper() in user_query.upper()), None)
            if target_well:
                self.log_capture.add(f"🎯 Target: {target_well}", "INFO")
            
            # Retrieve chunks
            retrieved = self.rag_agent.retrieve(user_query, mode=mode, well_name=target_well)
            self.log_capture.add(f"✓ Retrieved {len(retrieved['chunks'])} chunks (Hybrid)", "SUCCESS")
            
            # Mode-specific processing
            if mode == "extract":
                # Enhanced trajectory extraction
                from agents.parameter_extraction_agent import ParameterExtractionAgent
                ext = ParameterExtractionAgent(config)
                self.log_capture.add("🔧 Enhanced trajectory extraction...", "INFO")
                traj = ext.extract(retrieved['chunks'], self.log_capture)
                
                if traj['trajectory'] and len(traj['trajectory']) > 0:
                    self.last_trajectory = traj['trajectory']
                    answer = f"**Trajectory Extracted:** {len(traj['trajectory'])} points\n\n"
                    answer += f"**Confidence:** {traj.get('confidence', 0):.0%}\n"
                    answer += f"**Sources:** {traj.get('source_chunks', 0)} chunks analyzed\n\n"
                    answer += "| MD (m) | TVD (m) | ID (m) |\n|--------|---------|--------|\n"
                    
                    # Show first 20 points
                    for p in traj['trajectory'][:20]:
                        answer += f"| {p['MD']:.1f} | {p['TVD']:.1f} | {p.get('ID', 0.216):.3f} |\n"
                    
                    if len(traj['trajectory']) > 20:
                        answer += f"\n*Showing 20 of {len(traj['trajectory'])} points*"
                    
                    nodal_btn_state = gr.update(visible=True)
                    self.log_capture.add(f"✅ Extracted {len(traj['trajectory'])} points", "SUCCESS")
                else:
                    answer = "⚠️ No trajectory found in retrieved chunks.\n\n"
                    answer += "Try:\n• More specific query: 'Extract trajectory table from HAG-GT-01'\n"
                    answer += "• Check if document contains trajectory data"
                    nodal_btn_state = gr.update(visible=False)
                    self.log_capture.add("⚠️  No trajectory detected", "WARN")
            
            elif mode == 'summary':
                # Optimized summary mode - more chunks, better context
                context_parts = []
                for i, c in enumerate(retrieved['chunks'][:15]):  # Increased from 10 to 15
                    doc_name = Path(c['metadata'].get('source_file', 'Unknown')).stem
                    page = c['metadata'].get('page_number', '?')
                    
                    context_parts.append(
                        f"[{doc_name}, p.{page}]\n{c['content'][:400]}"  # More context per chunk
                    )
                
                context = '\n\n'.join(context_parts)
                
                prompt = f"""<|im_start|>system
Technical writer for geothermal well reports. Create structured summary (max 300 words).

CITATION RULES:
- Use exact document name + page
- Format: "2680 m (NLOG_GS_PUB_End-of-well-report-NAALDWIJK-GT-02-S1, p.7)"
- Cite after EVERY fact

Structure:
1. Well Name & Location
2. Depths (MD/TVD) with citations
3. Drilling Dates with citations
4. Key Formations
5. Completion Status

SI units only.
<|im_end|>
<|im_start|>user
{f"Context: {memory_context[:300]}" if memory_context else ""}

Question: {user_query}

Documents (15 chunks):
{context}
<|im_end|>
<|im_start|>assistant
**Well Summary:**
"""
                
                timeout = 900
                max_tokens = 450  # Slightly increased for better summaries
            
            else:
                # Factual Q&A mode
                context_parts = []
                for i, c in enumerate(retrieved['chunks'][:10]):
                    doc_name = Path(c['metadata'].get('source_file', 'Unknown')).stem
                    page = c['metadata'].get('page_number', '?')
                    
                    context_parts.append(
                        f"[{doc_name}, p.{page}]\n{c['content'][:700]}"
                    )
                
                context = '\n\n'.join(context_parts)
                
                prompt = f"""<|im_start|>system
Technical analyst. Answer with INLINE citations using exact document names.

RULES:
- Format: "2694 m (NLOG_GS_PUB_110211-EWOR-HAG-GT-01, p.8)"
- Cite after EVERY number/date
- SI units only
- Be precise

<|im_end|>
<|im_start|>user
{f"Previous: {memory_context[:300]}" if memory_context else ""}

Question: {user_query}

Documents:
{context}
<|im_end|>
<|im_start|>assistant
"""
                
                timeout = 300
                max_tokens = 300
            
            # Generate answer (if not extraction mode)
            if mode != 'extract':
                max_retries = 2
                answer = None
                
                for attempt in range(max_retries):
                    try:
                        self.log_capture.add(f"🤖 Qwen2.5:7b (attempt {attempt+1}/{max_retries})", "INFO")
                        
                        resp = requests.post(
                            f"{config['ollama']['host']}/api/generate",
                            json={
                                'model': config['ollama']['model_chat'],
                                'prompt': prompt,
                                'stream': False,
                                'options': {
                                    'temperature': 0.15,
                                    'num_ctx': 6144 if mode == 'summary' else 8192,
                                    'top_p': 0.9,
                                    'repeat_penalty': 1.1,
                                    'num_predict': max_tokens
                                }
                            },
                            timeout=timeout
                        )
                        
                        answer = resp.json().get('response', 'No response').strip()
                        self.log_capture.add("✅ Answer generated", "SUCCESS")
                        break
                    
                    except requests.Timeout:
                        if attempt < max_retries - 1:
                            self.log_capture.add(f"⏰ Timeout, retry...", "WARN")
                            timeout += 300
                            continue
                        else:
                            self.log_capture.add("❌ Final timeout", "ERROR")
                            return "⚠️ Timed out. Try simpler query or specify well name.", \
                                   "\n".join(self.log_capture.logs[-50:]), gr.update(visible=False)
                
                if not answer:
                    return "⚠️ No answer generated", "\n".join(self.log_capture.logs[-50:]), gr.update(visible=False)
                
                # Ensemble Judge Validation
                self.log_capture.add("⚖️  Ensemble validation...", "INFO")
                validation = self.judge.validate(answer, retrieved['chunks'], user_query)
                
                # Format validation results
                if hasattr(validation, '__getitem__') and 'ensemble_votes' in validation:
                    # Ensemble mode
                    votes = validation.get('ensemble_votes', [])
                    vote_summary = ', '.join([f"{v['model']}: {v['conf']:.0%}" for v in votes])
                    self.log_capture.add(f"🗳️  Votes: {vote_summary}", "INFO")
                
                if not validation['is_valid']:
                    self.log_capture.add(f"⚠️  Confidence: {validation['confidence']:.0%}", "WARN")
                    answer += f"\n\n### ⚠️ Validation\n"
                    answer += f"Confidence: {validation['confidence']:.0%}\n"
                    if validation.get('flagged_issues'):
                        answer += f"Issues: {', '.join(validation['flagged_issues'][:2])}"
                else:
                    self.log_capture.add(f"✅ Validated: {validation['confidence']:.0%}", "SUCCESS")
                
                nodal_btn_state = gr.update(visible=False)
            
            # Store in memory
            self.memory.add_turn(user_query, answer, {'well_name': target_well, 'mode': mode})
            self.log_capture.add("✅ Complete", "SUCCESS")
            
            return answer, "\n".join(self.log_capture.logs[-50:]), nodal_btn_state

        except Exception as e:
            import traceback
            self.log_capture.add(f"❌ ERROR: {str(e)[:200]}", "ERROR")
            return f"Error: {str(e)}\n{traceback.format_exc()[:300]}", \
                   "\n".join(self.log_capture.logs[-50:]), gr.update(visible=False)
    
    def run_nodal_analysis(self):
        """Execute nodal analysis with extracted trajectory"""
        if not self.last_trajectory:
            return "No trajectory data. Extract trajectory first."
        
        try:
            self.log_capture.add("🔧 Running Nodal Analysis...", "INFO")
            
            import json
            trajectory_file = 'temp_trajectory.json'
            
            with open(trajectory_file, 'w') as f:
                json.dump(self.last_trajectory, f, indent=2)
            
            self.log_capture.add(f"💾 Saved {len(self.last_trajectory)} points", "INFO")
            
            # Check if nodal analysis script exists
            nodal_script = Path('nodal/NodalAnalysis.py')
            if not nodal_script.exists():
                return "⚠️ nodal/NodalAnalysis.py not found. Create nodal analysis script first."
            
            result = subprocess.run(
                ['python', str(nodal_script), '--input', trajectory_file],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.log_capture.add("✅ Nodal analysis complete", "SUCCESS")
                return f"**Nodal Analysis Results:**\n\n{result.stdout}"
            else:
                return f"⚠️ Nodal analysis error:\n{result.stderr[:500]}"
        
        except subprocess.TimeoutExpired:
            return "⚠️ Nodal analysis timed out (>60s)"
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

# Initialize document store
store = DocumentStore()

# Gradio UI
with gr.Blocks(title="Geothermal RAG v5.0", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🌋 Geothermal RAG v5.0 - Ensemble Judge + Enhanced Trajectory")
    gr.Markdown("*Ultra-sophisticated chunking • Ensemble validation • Exact citations • Trajectory extraction*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Documents")
            files = gr.File(
                label="Select PDF Files", 
                file_count="multiple", 
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("📥 Index Files (60-120 sec)", variant="primary", size="lg")
            status = gr.Textbox(label="Upload Status", lines=4, interactive=False)
            
            gr.Markdown("### 📊 System Activity")
            log_box = gr.Textbox(label="Agent Logs", lines=10, interactive=False, show_copy_button=True)
            
            gr.Markdown("### 💡 Example Queries")
            gr.Markdown("""
- **Factual:** What is total depth of HAG-GT-01?
- **Summary:** Give a summary of NLW-GT-02-S1
- **Trajectory:** Extract trajectory of HAG-GT-01
- **Multi-turn:** What about the previous well?
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Your Question")
            query_text = gr.Textbox(
                label="Query", 
                lines=2, 
                placeholder="e.g., What is total depth of HAG-GT-01?"
            )
            
            with gr.Row():
                query_btn = gr.Button("🚀 Ask", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", size="sm")
            
            answer = gr.Textbox(
                label="Answer (Ensemble Validated)", 
                lines=16, 
                interactive=False, 
                show_copy_button=True
            )
            
            # Nodal analysis section
            nodal_btn = gr.Button(
                "▶️ Run Nodal Analysis", 
                visible=False, 
                variant="secondary"
            )
            nodal_output = gr.Textbox(
                label="Nodal Analysis Results", 
                lines=8, 
                visible=False,
                show_copy_button=True
            )
    
    # Event handlers
    upload_btn.click(
        fn=store.index_documents, 
        inputs=files, 
        outputs=[status, log_box]
    )
    
    query_btn.click(
        fn=store.query, 
        inputs=query_text, 
        outputs=[answer, log_box, nodal_btn]
    )
    
    query_text.submit(
        fn=store.query, 
        inputs=query_text, 
        outputs=[answer, log_box, nodal_btn]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", gr.update(visible=False)),
        outputs=[query_text, answer, nodal_btn]
    )
    
    nodal_btn.click(
        fn=store.run_nodal_analysis,
        outputs=nodal_output
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=nodal_output
    )

if __name__ == "__main__":
    print("="*80)
    print(" 🌋 GEOTHERMAL RAG v5.0 - PRODUCTION READY")
    print("="*80)
    print("✅ Ensemble validation (2 models)")
    print("✅ Enhanced trajectory extraction (regex + LLM)")
    print("✅ Better summaries (15 chunks)")
    print("✅ Ultra-sophisticated NLP chunking")
    print("✅ Exact document citations")
    print("✅ Multi-turn conversation memory")
    print("✅ Interactive nodal analysis")
    print("\n🌐 Starting server: http://127.0.0.1:7860")
    print("="*80)
    
    app.launch(
        server_name='127.0.0.1', 
        server_port=7860,
        show_error=True
    )
