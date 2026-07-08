import os
import uuid
from deepcollector.config.settings import AppConfig
from deepcollector.utils.initialization import initialize_apis, configure_llama_index
from deepcollector.core.agent import CatalogAgent

def run_project(project_name="Tempo", backend="GEMINI"):
    print(f"🚀 Starting {project_name} using {backend} backend...")
    
    os.environ["DEEPCOLLECTOR_LLM_BACKEND"] = backend
    
    config = AppConfig(VERBOSITY_LEVEL=2)
    config.CURRENT_PROJECT_ID = f"PROJ_{project_name.upper()}"
    config.CURRENT_PROJECT_NAME = project_name
    
    keys, models = initialize_apis(config)
    
    if backend != "LOCAL_VLLM":
        configure_llama_index(config, models, keys)
    
    # Running locally without Google Sheets connection for now
    gc = None 
    
    agent = CatalogAgent(config, gc, keys=keys, models=models)
    agent.job_id = f"JOB_{uuid.uuid4().hex[:6].upper()}"
    
    # We will uncomment this later when we are ready to actually run it!
    # agent.execute_workflow(mode="AGENT")
    
    return agent

if __name__ == "__main__":
    print("✅ Pipeline loaded successfully.")
