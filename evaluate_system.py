#!/usr/bin/env python3
"""
Evaluation script for Geothermal RAG v5.0
Tests core functionality without requiring Ollama or actual documents
"""
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SystemEvaluator:
    """Evaluates the Geothermal RAG system components"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'passed': 0,
            'failed': 0
        }
    
    def test_config_loading(self):
        """Test configuration file loading"""
        test_name = "Configuration Loading"
        try:
            config_path = Path('config/config_v5.yaml')
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            config = yaml.safe_load(open(config_path))
            
            # Validate required keys
            required_keys = ['system', 'ollama', 'agents', 'vector_store']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Missing required config key: {key}")
            
            # Check specific settings
            assert config['system']['name'] == 'Geothermal RAG v5'
            assert config['system']['version'] == '5.0'
            assert config['agents']['judge']['use_ensemble'] == True
            
            self.record_test(test_name, True, "Config loaded successfully")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def test_agent_imports(self):
        """Test that all agent modules can be imported"""
        test_name = "Agent Module Imports"
        try:
            # Import all agents
            from agents.chat_memory import ChatMemory
            from agents.ensemble_judge_agent import EnsembleJudgeAgent
            from agents.fact_checking_agent import FactCheckingAgent
            from agents.ingestion_agent import IngestionAgent
            from agents.judge_agent import JudgeAgent
            from agents.parameter_extraction_agent import ParameterExtractionAgent
            from agents.preprocessing_agent import PreprocessingAgent
            from agents.rag_retrieval_agent import RAGRetrievalAgent
            
            self.record_test(test_name, True, "All agents imported successfully")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        test_name = "Directory Structure"
        try:
            required_dirs = ['agents', 'config', 'data']
            missing = []
            
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing.append(dir_name)
            
            if missing:
                raise FileNotFoundError(f"Missing directories: {', '.join(missing)}")
            
            self.record_test(test_name, True, "All required directories exist")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def test_app_structure(self):
        """Test that main app file is properly structured"""
        test_name = "Application Structure"
        try:
            app_path = Path('app_working.py')
            if not app_path.exists():
                raise FileNotFoundError("app_working.py not found")
            
            # Read and check for key components
            content = app_path.read_text()
            
            required_components = [
                'import gradio',
                'class DocumentStore',
                'def index_documents',
                'def query',
                'def run_nodal_analysis',
                'app.launch'
            ]
            
            missing = []
            for component in required_components:
                if component not in content:
                    missing.append(component)
            
            if missing:
                raise ValueError(f"Missing components: {', '.join(missing)}")
            
            self.record_test(test_name, True, "App structure validated")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def test_dependencies(self):
        """Test that required dependencies are installed"""
        test_name = "Python Dependencies"
        try:
            import gradio
            import chromadb
            import fitz  # PyMuPDF
            import pdfplumber
            import yaml
            import requests
            
            # Check versions
            import pkg_resources
            packages = {
                'gradio': '4.13.0',
                'chromadb': '0.5.5',
                'PyMuPDF': '1.24.10',
                'pdfplumber': '0.11.4'
            }
            
            version_info = []
            for pkg, expected in packages.items():
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    version_info.append(f"{pkg}=={version}")
                except:
                    version_info.append(f"{pkg}==unknown")
            
            self.record_test(test_name, True, f"Dependencies OK: {', '.join(version_info)}")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def test_config_validation(self):
        """Test configuration values are valid"""
        test_name = "Configuration Validation"
        try:
            config = yaml.safe_load(open('config/config_v5.yaml'))
            
            # Test embedding strategies
            strategies = config['embedding_strategies']
            assert 'factual_qa' in strategies
            assert 'technical_extraction' in strategies
            assert 'summary' in strategies
            
            # Test agent configurations
            assert config['agents']['rag_retrieval']['top_k_factual'] == 10
            assert config['agents']['judge']['use_ensemble'] == True
            assert len(config['agents']['judge']['ensemble_models']) == 2
            
            # Test Ollama config
            assert config['ollama']['host'] == 'http://localhost:11434'
            assert config['ollama']['model_chat'] == 'qwen2.5:7b'
            
            self.record_test(test_name, True, "Configuration values validated")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def test_memory_system(self):
        """Test conversation memory system"""
        test_name = "Memory System"
        try:
            config = yaml.safe_load(open('config/config_v5.yaml'))
            from agents.chat_memory import ChatMemory
            
            memory = ChatMemory(config)
            
            # Test adding conversation turns
            memory.add_turn(
                "What is the depth of HAG-GT-01?",
                "The total depth is 2694 m (TVD)",
                {'well_name': 'HAG-GT-01', 'mode': 'qa'}
            )
            
            # Test context retrieval
            context = memory.get_context("Tell me more about that well")
            assert context is not None
            
            self.record_test(test_name, True, "Memory system operational")
            logger.info(f"✅ {test_name}: PASSED")
            return True
            
        except Exception as e:
            self.record_test(test_name, False, str(e))
            logger.error(f"❌ {test_name}: FAILED - {e}")
            return False
    
    def record_test(self, name, passed, details):
        """Record test result"""
        self.results['tests'].append({
            'name': name,
            'passed': passed,
            'details': details
        })
        if passed:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
    
    def run_all_tests(self):
        """Run all evaluation tests"""
        logger.info("=" * 80)
        logger.info(" 🧪 GEOTHERMAL RAG v5.0 - SYSTEM EVALUATION")
        logger.info("=" * 80)
        logger.info("")
        
        tests = [
            self.test_directory_structure,
            self.test_config_loading,
            self.test_config_validation,
            self.test_dependencies,
            self.test_agent_imports,
            self.test_app_structure,
            self.test_memory_system,
        ]
        
        for test in tests:
            test()
            logger.info("")
        
        return self.results
    
    def print_summary(self):
        """Print evaluation summary"""
        logger.info("=" * 80)
        logger.info(" 📊 EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests: {len(self.results['tests'])}")
        logger.info(f"Passed: {self.results['passed']} ✅")
        logger.info(f"Failed: {self.results['failed']} ❌")
        logger.info(f"Success Rate: {(self.results['passed']/len(self.results['tests'])*100):.1f}%")
        logger.info("=" * 80)
        
        if self.results['failed'] == 0:
            logger.info("🎉 ALL TESTS PASSED! System is ready for deployment.")
        else:
            logger.warning("⚠️  Some tests failed. Review the errors above.")
        
        return self.results['failed'] == 0


def main():
    """Main evaluation function"""
    evaluator = SystemEvaluator()
    evaluator.run_all_tests()
    success = evaluator.print_summary()
    
    # Write results to file
    import json
    results_file = Path('data/evaluation_results.json')
    results_file.parent.mkdir(exist_ok=True)
    results_file.write_text(json.dumps(evaluator.results, indent=2))
    logger.info(f"\n📄 Detailed results saved to: {results_file}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
