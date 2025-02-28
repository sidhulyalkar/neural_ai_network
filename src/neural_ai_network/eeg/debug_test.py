# debug_test.py
import os
import sys
import logging
import json
import traceback

def setup_logging():
    """Set up detailed logging for debugging."""
    logger = logging.getLogger("DebugTest")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler("debug_log.txt")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def test_imports(logger):
    """Test importing the required modules."""
    logger.info("Testing imports...")
    
    try:
        logger.debug("Importing numpy...")
        import numpy as np
        logger.debug("Numpy version: " + np.__version__)
        
        logger.debug("Importing mne...")
        import mne
        logger.debug("MNE version: " + mne.__version__)
        
        try:
            logger.debug("Testing mne.time_frequency imports...")
            from mne.time_frequency import psd_welch
            logger.debug("mne.time_frequency.psd_welch exists")
        except ImportError:
            logger.debug("mne.time_frequency.psd_welch not found")
            try:
                from mne.time_frequency import psd_array_welch
                logger.debug("mne.time_frequency.psd_array_welch exists")
            except ImportError:
                logger.debug("mne.time_frequency.psd_array_welch not found")
        
        try:
            logger.debug("Testing raw.compute_psd...")
            # Create a minimal raw object
            data = np.random.random((2, 1000))
            info = mne.create_info(ch_names=['ch1', 'ch2'], sfreq=100, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # Test compute_psd
            if hasattr(raw, 'compute_psd'):
                logger.debug("raw.compute_psd exists")
                psds = raw.compute_psd(fmin=0, fmax=50)
                logger.debug("raw.compute_psd works")
            else:
                logger.debug("raw.compute_psd does not exist")
        except Exception as e:
            logger.debug(f"Error testing compute_psd: {e}")
        
        logger.debug("Import neural_ai_network...")
        from neural_ai_network.eeg.eeg_agent import EEGProcessingAgent
        logger.info("All imports successful!")
        return True
    except Exception as e:
        logger.error(f"Import error: {e}")
        logger.error(traceback.format_exc())
        return False

def test_data_loading(logger, file_path):
    """Test loading a file with MNE."""
    logger.info(f"Testing data loading for file: {file_path}")
    
    try:
        import mne
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        logger.debug(f"File extension: {ext}")
        
        # Check if epoch file
        is_epoch_file = "-epo.fif" in file_path or "_epo.fif" in file_path
        logger.debug(f"Is epoch file: {is_epoch_file}")
        
        if is_epoch_file:
            logger.debug("Loading as epochs...")
            try:
                epochs = mne.read_epochs(file_path, preload=True)
                logger.info(f"Successfully loaded epochs: {len(epochs)} epochs, {len(epochs.ch_names)} channels")
                return True
            except Exception as e:
                logger.error(f"Error loading epochs: {e}")
                logger.error(traceback.format_exc())
                return False
        else:
            logger.debug("Loading as raw...")
            try:
                if ext == '.edf':
                    raw = mne.io.read_raw_edf(file_path, preload=True)
                elif ext == '.bdf':
                    raw = mne.io.read_raw_bdf(file_path, preload=True)
                elif ext in ['.fif', '.fiff']:
                    raw = mne.io.read_raw_fif(file_path, preload=True)
                elif ext == '.vhdr':
                    raw = mne.io.read_raw_brainvision(file_path, preload=True)
                elif ext == '.set':
                    raw = mne.io.read_raw_eeglab(file_path, preload=True)
                else:
                    logger.error(f"Unsupported file format: {ext}")
                    return False
                
                logger.info(f"Successfully loaded raw data: {len(raw.ch_names)} channels, {raw.n_times} samples at {raw.info['sfreq']} Hz")
                return True
            except Exception as e:
                logger.error(f"Error loading raw data: {e}")
                logger.error(traceback.format_exc())
                return False
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        logger.error(traceback.format_exc())
        return False

def test_eeg_agent_init(logger):
    """Test initializing the EEG agent."""
    logger.info("Testing EEG agent initialization...")
    
    # First, ensure the config file exists
    if not os.path.exists("eeg_agent_config.json"):
        logger.info("Creating default config file")
        with open("eeg_agent_config.json", 'w') as f:
            json.dump({
                "message_broker": {
                    "host": "localhost",
                    "port": 5672,
                    "username": "guest",
                    "password": "guest",
                    "queue": "eeg_processing"
                },
                "preprocessing": {
                    "filter": {
                        "highpass": 1.0,
                        "lowpass": 40.0
                    },
                    "notch": 60.0,
                    "reference": "average"
                },
                "storage": {
                    "processed_data": "./test_output",
                    "results": "./test_output"
                }
            }, f, indent=4)
    
    try:
        from neural_ai_network.eeg.eeg_agent import EEGProcessingAgent
        logger.debug("Initializing EEGProcessingAgent...")
        agent = EEGProcessingAgent("eeg_agent_config.json")
        logger.info("Successfully initialized EEG agent")
        return agent
    except Exception as e:
        logger.error(f"Error initializing EEG agent: {e}")
        logger.error(traceback.format_exc())
        return None

def test_psd_function(logger):
    """Test the PSD function directly."""
    logger.info("Testing PSD function...")
    
    try:
        import mne
        import numpy as np
        
        # Create a test raw object
        n_channels = 3
        n_samples = 1000
        sfreq = 100
        
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(ch_names=[f'ch{i}' for i in range(n_channels)], 
                              sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        logger.debug("Raw object created successfully")
        
        # Try different PSD methods
        try:
            logger.debug("Trying raw.compute_psd()...")
            psds, freqs = raw.compute_psd(fmin=0.5, fmax=50).get_data(return_freqs=True)
            logger.info("raw.compute_psd() succeeded")
            logger.debug(f"PSD shape: {psds.shape}, freqs shape: {freqs.shape}")
            return "compute_psd"
        except Exception as e1:
            logger.debug(f"raw.compute_psd() failed: {e1}")
            
            try:
                logger.debug("Trying mne.time_frequency.psd_welch()...")
                from mne.time_frequency import psd_welch
                psds, freqs = psd_welch(raw, fmin=0.5, fmax=50)
                logger.info("mne.time_frequency.psd_welch() succeeded")
                logger.debug(f"PSD shape: {psds.shape}, freqs shape: {freqs.shape}")
                return "psd_welch"
            except Exception as e2:
                logger.debug(f"mne.time_frequency.psd_welch() failed: {e2}")
                
                try:
                    logger.debug("Trying mne.time_frequency.psd_array_welch()...")
                    from mne.time_frequency import psd_array_welch
                    psds, freqs = psd_array_welch(data, sfreq=sfreq, fmin=0.5, fmax=50)
                    logger.info("mne.time_frequency.psd_array_welch() succeeded")
                    logger.debug(f"PSD shape: {psds.shape}, freqs shape: {freqs.shape}")
                    return "psd_array_welch"
                except Exception as e3:
                    logger.debug(f"mne.time_frequency.psd_array_welch() failed: {e3}")
                    logger.error("All PSD methods failed")
                    return None
    except Exception as e:
        logger.error(f"Error in PSD function test: {e}")
        logger.error(traceback.format_exc())
        return None

def main():
    logger = setup_logging()
    logger.info("Starting debug test")
    
    # Test importing modules
    if not test_imports(logger):
        logger.error("Import test failed, exiting")
        return
    
    # Test PSD function
    psd_method = test_psd_function(logger)
    if not psd_method:
        logger.error("PSD function test failed, exiting")
        return
    logger.info(f"Will use PSD method: {psd_method}")
    
    # Test EEG agent initialization
    agent = test_eeg_agent_init(logger)
    if not agent:
        logger.error("EEG agent initialization failed, exiting")
        return
    
    # If a file path is provided, test loading it
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not test_data_loading(logger, file_path):
            logger.error("Data loading test failed, exiting")
            return
        
        # If all tests passed, try processing the file
        logger.info(f"Beginning full processing of {file_path}")
        try:
            logger.debug("Calling agent.process_data()...")
            result = agent.process_data(file_path)
            logger.info(f"Processing result: {result}")
        except Exception as e:
            logger.error(f"Error in process_data: {e}")
            logger.error(traceback.format_exc())
    
    logger.info("Debug test completed")

if __name__ == "__main__":
    main()