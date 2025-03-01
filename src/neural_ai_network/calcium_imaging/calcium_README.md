# Calcium Imaging Module for Neural AI Network

This module provides tools for processing and analyzing calcium imaging data within the Neural AI Network framework. It includes a specialized agent for calcium imaging data processing, a data loader for various calcium imaging datasets, and utilities for visualization and analysis.

## Components

### 1. Calcium Processing Agent (`calcium_agent.py`)

A specialized agent for processing calcium imaging data with the following capabilities:

- Loading various calcium imaging data formats (TIFF, HDF5, numpy arrays)
- Preprocessing: motion correction, spatial filtering, temporal filtering, background removal
- Cell detection: watershed-based segmentation, support for CNMF and Suite2p methods
- Signal extraction: ROI-based signal extraction with neuropil correction
- Event detection: identifying calcium transients in extracted signals
- Analysis: cell-level and population-level statistics, activity patterns
- Visualization: cell maps, activity traces, event detection

The agent can be used independently or as part of the Neural Orchestrator system.

### 2. Calcium Data Loader (`calcium_data_loader.py`)

A utility for loading and managing calcium imaging datasets:

- Support for Neurofinder challenge datasets
- Support for Allen Brain Observatory data (via AllenSDK)
- Support for CRCNS calcium imaging datasets
- Small sample datasets for testing
- Utilities for downloading and preprocessing datasets

### 3. Test Script (`test_calcium_agent.py`)

A script for testing the calcium agent's functionality:

- Tests with sample data from Neurofinder
- Creates synthetic calcium imaging data for testing
- Demonstrates integration with the Neural Orchestrator
- Showcases analysis capabilities

## Installation

1. Install the required dependencies:

```bash
pip install numpy scipy scikit-image h5py pandas matplotlib tqdm requests pika
```

2. Optional dependencies for advanced functionality:

```bash
# For Allen Brain Observatory data
pip install allensdk

# For advanced cell detection methods (CNMF)
pip install caiman

# For Suite2p integration
pip install suite2p
```

3. Place the module files in the neural_ai_network package structure:

```
neural_ai_network/
├── core/
│   ├── neural_orchestrator.py
│   └── ...
├── calcium/
│   ├── calcium_agent.py
│   ├── calcium_data_loader.py
│   └── preprocessing.py (optional)
└── ...
```

## Usage

### Basic Usage

```python
from neural_ai_network.calcium.calcium_agent import CalciumProcessingAgent
from neural_ai_network.calcium.calcium_data_loader import CalciumDataLoader

# Initialize the data loader and agent
loader = CalciumDataLoader()
agent = CalciumProcessingAgent()

# Load a sample dataset
sample_data = loader.load_sample_dataset("sample1")
data = sample_data["data"]

# Process the data
result = agent.process_data("/path/to/your/calcium_data.tif")

# Access the results
print(f"Cells detected: {result.get('cells_detected', 0)}")
print(f"Events detected: {result.get('events_detected', 0)}")
```

### Integration with Neural Orchestrator

```python
from neural_ai_network.core.neural_orchestrator import NeuralDataOrchestrator

# Initialize the orchestrator
orchestrator = NeuralDataOrchestrator()

# Process calcium imaging data
job_id = orchestrator.process_data("/path/to/your/calcium_data.tif", 
                                  modality="calcium_agent")

# Get job status
status = orchestrator.get_job_status(job_id)
print(f"Job status: {status}")
```

## Configuration

The calcium agent behavior can be customized through a configuration file or by passing parameters:

```python
# Custom processing parameters
parameters = {
    "preprocessing": {
        "spatial_filter": "gaussian",
        "temporal_filter": "savgol",
        "motion_correction": "ecc",
        "background_removal": "percentile"
    },
    "cell_detection": {
        "method": "watershed",
        "min_cell_size": 30,
        "max_cell_size": 500
    },
    # Additional parameters...
}

# Process with custom parameters
result = agent.process_data("/path/to/your/data.tif", parameters=parameters)
```

## Available Datasets

### Neurofinder Datasets

The Neurofinder challenge provides calcium imaging datasets with human-annotated ground truth cells:

- Neurofinder 00.00, 00.01: GCaMP6s, high SNR, widefield, cultured neurons
- Neurofinder 01.00, 01.01: GCaMP6s, high SNR, two-photon, visual cortex
- Neurofinder 02.00, 02.01: GCaMP6s, low/medium SNR, two-photon, visual cortex
- Neurofinder 03.00: GCaMP6f, low SNR, two-photon, visual cortex
- Neurofinder 04.00, 04.01: GCaMP5k, low SNR, two-photon, visual cortex

To download and use:

```python
loader = CalciumDataLoader()
dataset = loader.load_neurofinder_dataset("00.00")
```

### Allen Brain Observatory

The Allen Institute's Brain Observatory provides a large-scale, standardized survey of physiological activity in the mouse visual cortex, featuring calcium imaging data:

```python
loader = CalciumDataLoader()
allen_data = loader.load_allen_brain_data()
```

Note: This requires the AllenSDK to be installed.

### CRCNS Datasets

The Collaborative Research in Computational Neuroscience (CRCNS) data sharing initiative includes several calcium imaging datasets:

- cai-1: Calcium imaging of mouse V1 responses to oriented gratings
- cai-2: Calcium imaging of mouse barrel cortex during pole localization 
- cai-3: Calcium imaging and simultaneous electrophysiology in visual cortex

These datasets require registration at crcns.org.

## Output Files

The calcium agent generates the following output files:

- `cell_masks.npy`: Binary masks for detected cells
- `std_projection.npy`, `max_projection.npy`: Summary projections
- `raw_signals.npy`, `df_f_signals.npy`: Extracted calcium signals
- `events.json`: Detected calcium events
- `analysis.json`: Analysis results and statistics
- `cell_summary.json`: Summary of cell properties
- Visualization images (PNG format): cell maps, example traces, etc.

## Examples

See the `test_calcium_agent.py` script for a complete example of using the module.