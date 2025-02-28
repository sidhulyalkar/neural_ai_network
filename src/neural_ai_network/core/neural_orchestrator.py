# neural_orchestrator.py
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pika  # For RabbitMQ messaging

class NeuralDataOrchestrator:
    """
    Central orchestration system for coordinating neural data processing agents.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.agents = {}
        self.connection = None
        self.channel = None
        self.active_jobs = {}  # Track active jobs
        
        # Initialize message broker connection
        self._setup_message_broker()
        
        # Register available agents
        self._register_agents()
        
        self.logger.info("Neural Data Orchestrator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("NeuralOrchestrator")
        logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if logger already exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Add file handler for persistent logs
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/orchestrator.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                "message_broker": {
                    "host": "localhost",
                    "port": 5672,
                    "username": "guest",
                    "password": "guest"
                },
                "agents": {
                    "eeg_agent": {"queue": "eeg_processing", "supported_formats": [".edf", ".bdf"]},
                    "fmri_agent": {"queue": "fmri_processing", "supported_formats": [".nii", ".nii.gz"]},
                    "calcium_agent": {"queue": "calcium_processing", "supported_formats": [".tif", ".tiff"]}
                },
                "storage": {
                    "raw_data": "./data/raw",
                    "processed_data": "./data/processed",
                    "results": "./data/results"
                }
            }
    
    def _setup_message_broker(self):
        """Set up connection to message broker (RabbitMQ)."""
        try:
            broker_config = self.config["message_broker"]
            credentials = pika.PlainCredentials(
                broker_config.get("username", "guest"),
                broker_config.get("password", "guest")
            )
            parameters = pika.ConnectionParameters(
                host=broker_config.get("host", "localhost"),
                port=broker_config.get("port", 5672),
                credentials=credentials,
                heartbeat=600,  # Longer heartbeat for stability
                blocked_connection_timeout=300
            )
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Declare queues for each agent
            for agent_name, agent_config in self.config["agents"].items():
                queue_name = agent_config["queue"]
                self.channel.queue_declare(queue=queue_name, durable=True)
                
                # Declare result queue for each agent
                result_queue = f"{queue_name}_results"
                self.channel.queue_declare(queue=result_queue, durable=True)
            
            # Declare a status queue for tracking job statuses
            self.channel.queue_declare(queue="job_status", durable=True)
            
            self.logger.info("Message broker connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to message broker: {e}")
            self.logger.warning("Operating in local-only mode (no distributed processing)")
    
    def _register_agents(self):
        """Register available agents from configuration."""
        for agent_name, agent_config in self.config["agents"].items():
            self.agents[agent_name] = agent_config
            self.logger.info(f"Registered agent: {agent_name}")
    
    def process_data(self, data_path: str, modality: Optional[str] = None, 
                     parameters: Optional[Dict] = None) -> str:
        """
        Process neural data by routing to appropriate agent.
        
        Args:
            data_path: Path to data file or directory
            modality: Optional explicit modality to use
            parameters: Optional processing parameters
            
        Returns:
            job_id: Identifier for the processing job
        """
        # Generate a proper UUID for the job
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Determine appropriate agent based on file extension or explicit modality
        agent_name = self._determine_agent(data_path, modality)
        
        if not agent_name:
            self.logger.error(f"No suitable agent found for {data_path}")
            return None
        
        # Prepare job message
        timestamp = datetime.now().isoformat()
        message = {
            "job_id": job_id,
            "data_path": data_path,
            "timestamp": timestamp,
            "parameters": parameters or {}
        }
        
        # Track job in active jobs
        self.active_jobs[job_id] = {
            "agent": agent_name,
            "status": "queued",
            "timestamp": timestamp,
            "data_path": data_path
        }
        
        # Send to appropriate queue if message broker is available
        if self.channel:
            queue_name = self.agents[agent_name]["queue"]
            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                )
            )
            self.logger.info(f"Job {job_id} sent to {agent_name} via queue {queue_name}")
            
            # Publish job status to status queue
            self.channel.basic_publish(
                exchange='',
                routing_key="job_status",
                body=json.dumps({
                    "job_id": job_id,
                    "status": "queued",
                    "agent": agent_name,
                    "timestamp": timestamp
                }),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                )
            )
        else:
            # Local processing (for development/testing)
            self.logger.info(f"Would process {data_path} with {agent_name} (local mode)")
        
        return job_id
    
    def _determine_agent(self, data_path: str, explicit_modality: Optional[str] = None) -> Optional[str]:
        """
        Determine which agent should process the given data.
        
        Args:
            data_path: Path to data file
            explicit_modality: Optional explicit modality specification
            
        Returns:
            agent_name: Name of suitable agent or None if no match
        """
        if explicit_modality and explicit_modality in self.agents:
            return explicit_modality
        
        # Determine by file extension
        _, extension = os.path.splitext(data_path.lower())
        for agent_name, agent_config in self.agents.items():
            if extension in agent_config["supported_formats"]:
                return agent_name
        
        # If no match by extension, try to infer from content
        # This is a placeholder for more sophisticated content-based detection
        if os.path.exists(data_path) and os.path.isfile(data_path):
            # Read first few bytes to determine file type
            try:
                with open(data_path, 'rb') as f:
                    header = f.read(64)
                
                # Simple checks for common neural data formats
                if header.startswith(b'NIFTI'):
                    return 'fmri_agent'
                elif b'EDF+' in header:
                    return 'eeg_agent'
                elif header.startswith(b'BrainVision'):
                    return 'eeg_agent'
            except Exception as e:
                self.logger.debug(f"Error reading file header: {e}")
        
        return None
    
    def get_job_status(self, job_id: str) -> Dict:
        """
        Get status of a processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            status: Dictionary with job status information
        """
        # Check local tracking first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # TODO: Implement more robust job status tracking (e.g., via database)
        return {"job_id": job_id, "status": "unknown"}
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict]:
        """
        List all jobs with optional filtering by status.
        
        Args:
            status: Optional status to filter by
            
        Returns:
            List of job information dictionaries
        """
        if status:
            return [job for job_id, job in self.active_jobs.items() 
                   if job["status"] == status]
        return list(self.active_jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Attempt to cancel a job if it hasn't started processing.
        
        Args:
            job_id: Job identifier
            
        Returns:
            bool: Whether the cancellation was successful
        """
        if job_id in self.active_jobs and self.active_jobs[job_id]["status"] == "queued":
            self.active_jobs[job_id]["status"] = "cancelled"
            
            # Update job status in message broker if available
            if self.channel:
                self.channel.basic_publish(
                    exchange='',
                    routing_key="job_status",
                    body=json.dumps({
                        "job_id": job_id,
                        "status": "cancelled",
                        "timestamp": datetime.now().isoformat()
                    }),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                    )
                )
            
            self.logger.info(f"Cancelled job: {job_id}")
            return True
        
        self.logger.warning(f"Cannot cancel job {job_id}: already processing or completed")
        return False
    
    def register_result_callback(self, callback):
        """
        Register a callback function to be called when job results are received.
        
        Args:
            callback: Function to call with job results
        """
        if not self.channel:
            self.logger.warning("No message broker connection, cannot register callback")
            return
        
        # Set up consumer for each agent's result queue
        for agent_name, agent_config in self.agents.items():
            result_queue = f"{agent_config['queue']}_results"
            
            def result_callback(ch, method, properties, body):
                try:
                    result = json.loads(body)
                    job_id = result.get("job_id")
                    
                    if job_id in self.active_jobs:
                        self.active_jobs[job_id]["status"] = "completed"
                        self.active_jobs[job_id]["result"] = result
                    
                    # Call user callback
                    callback(result)
                    
                    # Acknowledge message
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    self.logger.error(f"Error in result callback: {e}")
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
            self.channel.basic_consume(
                queue=result_queue,
                on_message_callback=result_callback
            )
        
        self.logger.info("Registered result callback for all agent queues")
    
    def shutdown(self):
        """Clean shutdown of the orchestrator."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.info("Message broker connection closed")
            except Exception as e:
                self.logger.error(f"Error closing connection: {e}")
        self.logger.info("Neural Data Orchestrator shut down")


# Example usage
if __name__ == "__main__":
    orchestrator = NeuralDataOrchestrator()
    
    # Example: Process an EEG file
    job_id = orchestrator.process_data("sample_data/subject001.edf")
    print(f"Started job: {job_id}")
    
    # Explicit modality specification
    job_id = orchestrator.process_data("sample_data/calcium_recording.raw", modality="calcium_agent")
    print(f"Started job with explicit modality: {job_id}")
    
    # Example with processing parameters
    params = {
        "preprocessing": {
            "filter": {
                "highpass": 0.5,
                "lowpass": 30.0
            },
            "notch": 50.0
        },
        "analysis": {
            "features": ["bandpower", "connectivity"]
        }
    }
    job_id = orchestrator.process_data("sample_data/subject002.edf", parameters=params)
    print(f"Started job with custom parameters: {job_id}")
    
    # Check job status
    status = orchestrator.get_job_status(job_id)
    print(f"Job status: {status}")
    
    orchestrator.shutdown()