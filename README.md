# Neural AI Network

A distributed neural data processing system for orchestrating and processing multi-modal neural recordings.

## Overview

Neural AI Network is a flexible framework designed to process and analyze various types of neural recording data, including EEG, fMRI, calcium imaging, and more. The system uses a microservices architecture with a central orchestrator that coordinates specialized processing agents for different neural data modalities.

## System Architecture

```
                           ┌─────────────┐
                           │             │
                           │Orchestrator │
                           │             │
                           └──────┬──────┘
                                  │
                                  │
               ┌─────────────────┼─────────────────┐
               │                 │                 │
        ┌──────▼─────┐    ┌──────▼─────┐    ┌──────▼─────┐
        │            │    │            │    │            │
        │ EEG Agent  │    │ fMRI Agent │    │Calcium Img │
        │            │    │            │    │   Agent    │
        └────────────┘    └────────────┘    └────────────┘
```

### Core Components

1. **Neural Data Orchestrator**: Central coordination system that routes data to appropriate agents based on file types and orchestrates processing pipelines.

2. **Specialized Agents**: Modality-specific processing units that handle different types of neural data:
   - EEG Processing Agent
   - fMRI Processing Agent
   - Calcium Imaging Agent
   - (More agents to be added)

3. **Message Broker**: RabbitMQ-based communication system enabling asynchronous processing and distributed deployment.

4. **Common Utilities**:
   - Data loaders for each modality
   - Preprocessing pipelines
   - Feature extraction frameworks

## Supported Neural Data Modalities

### EEG (Electroencephalography)

EEG measures electrical activity of the brain via electrodes placed on the scalp.

**Features:**
- Preprocessing: Filtering, artifact removal, bad channel detection, re-referencing
- Feature extraction: Band powers, connectivity metrics, time-domain features
- Support for multiple file formats: EDF, BDF, SET (EEGLAB), FIF (MNE)
- Time-warp invariant pattern detection for identifying ERP-like components
- Integration with Temple University EEG Corpus

### fMRI (Functional Magnetic Resonance Imaging)

fMRI measures brain activity by detecting changes in blood flow, using the BOLD (Blood Oxygen Level Dependent) signal.

**Features:**
- Support for NIfTI format (.nii, .nii.gz)
- Preprocessing: Motion correction, spatial normalization, smoothing
- Analysis: GLM, ICA, functional connectivity

### Calcium Imaging

Calcium imaging visualizes neuron activity by measuring calcium concentration changes in neurons using fluorescent indicators.

**Features:**
- Support for TIFF stacks (.tif, .tiff)
- Cell identification and signal extraction
- Deconvolution of calcium signals to infer spike timing
- ROI management and visualization

## Installation and Setup

### Prerequisites

- Python 3.8+
- RabbitMQ server (for distributed processing)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-ai-network.git
cd neural-ai-network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the system:
```bash
cp config.json.example config.json
# Edit config.json with your settings
```

## Usage

### Starting the Orchestrator

```bash
python -m neural_ai_network.core.neural_orchestrator
```

### Starting Agents

```bash
# Start EEG agent
python -m neural_ai_network.eeg.eeg_agent

# Start fMRI agent
python -m neural_ai_network.fmri.fmri_agent

# Start Calcium Imaging agent
python -m neural_ai_network.calcium.calcium_agent
```

### Processing Data

```python
from neural_ai_network.core.neural_orchestrator import NeuralDataOrchestrator

# Initialize orchestrator
orchestrator = NeuralDataOrchestrator()

# Process an EEG file
job_id = orchestrator.process_data("path/to/eeg_recording.edf")
print(f"Job started: {job_id}")

# Process with explicit modality
job_id = orchestrator.process_data("path/to/data.raw", modality="calcium_agent")
```

## Code Structure

```
neural-ai-network/
├── neural_ai_network/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── neural_orchestrator.py  # Central orchestration system
│   │   └── utils.py                # Shared utilities
│   ├── eeg/
│   │   ├── __init__.py
│   │   ├── eeg_agent.py            # EEG processing agent
│   │   ├── preprocessing.py        # EEG preprocessing pipeline
│   │   └── data_loader.py          # EEG data loading utilities
│   ├── fmri/
│   │   ├── __init__.py
│   │   ├── fmri_agent.py           # fMRI processing agent
│   │   └── preprocessing.py        # fMRI preprocessing pipeline
│   ├── calcium/
│   │   ├── __init__.py
│   │   └── calcium_agent.py        # Calcium imaging processing agent
│   └── __init__.py
├── tests/
├── config.json                     # System configuration
├── requirements.txt
└── README.md
```

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

Or test specific components:

```bash
python -m tests.test_eeg_agent
```

## References and Resources

- [MNE-Python](https://mne.tools/stable/index.html) - Used for EEG/MEG data processing
- [Temple University EEG Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/) - Clinical EEG dataset
- [Nipy](https://nipy.org/) - Neuroimaging in Python
- [CaImAn](https://github.com/flatironinstitute/CaImAn) - Calcium Imaging Analysis

## License

[MIT License](LICENSE)

## Contributors

- Sidharth Hulyalkar - Initial work, testing, and maintenance