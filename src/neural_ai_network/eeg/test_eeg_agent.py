# test_eeg_agent.py
import os
import json
import glob
from neural_ai_network.eeg.eeg_agent import EEGProcessingAgent
from neural_ai_network.core import NeuralDataOrchestrator

def test_eeg_agent():
    """Test the EEG agent with processed data."""
    print("Testing EEG Agent with processed data")
    
    # Initialize the agent
    agent = EEGProcessingAgent()
    
    # Find processed data
    data_dir = "./demos/output/openneuro_eeg"
    results_files = []
    
    # Look for results.json files in the output directory
    for dataset_id in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_id)
        if os.path.isdir(dataset_path):
            for file_dir in glob.glob(os.path.join(dataset_path, "file_*")):
                results_path = os.path.join(file_dir, "results", "results.json")
                if os.path.exists(results_path):
                    results_files.append(results_path)
    
    print(f"Found {len(results_files)} processed EEG files")
    
    # Test agent with each file
    for result_file in results_files:
        print(f"\nAnalyzing: {result_file}")
        
        # Load results
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Ask agent to analyze the results
        analysis = agent.analyze_features(results["features"])
        
        # Print analysis
        print("\nEEG AGENT ANALYSIS:")
        print("-" * 40)
        print(analysis)
        
        # Test more specific queries
        if "bandpower" in results["features"]:
            alpha_analysis = agent.analyze_band_activity(
                results["features"]["bandpower"], 
                band="alpha"
            )
            print("\nALPHA ACTIVITY ANALYSIS:")
            print("-" * 40)
            print(alpha_analysis)

if __name__ == "__main__":
    test_eeg_agent()