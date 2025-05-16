# API Reference

This page provides detailed documentation for the AutoAI-AgentRAG API.

## Agent API

### Agent Class

```python
class Agent:
    """
    The core Agent class for building intelligent automation agents.
    
    Agents can be configured with different capabilities, connect to RAG systems,
    and utilize ML models to perform complex tasks.
    """
    
    def __init__(
        self, 
        name: str, 
        agent_type: AgentType = AgentType.TASK_ORIENTED,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize a new Agent instance.
        
        Args:
            name: A human-readable name for the agent
            agent_type: The type of agent functionality to enable
            config: Optional configuration parameters
        """
        pass
        
    def connect_rag(self, rag_pipeline: RAGPipeline) -> None:
        """
        Connect a RAG pipeline to this agent for knowledge retrieval.
        
        Args:
            rag_pipeline: The RAG pipeline to connect to this agent
        """
        pass
    
    def add_model(self, model: MLModel, model_name: Optional[str] = None) -> None:
        """
        Add a machine learning model to this agent.
        
        Args:
            model: The machine learning model to add
            model_name: Optional custom name for the model. If not provided, 
                       the model's default name will be used.
        """
        pass
    
    def execute(self, task: Union[str, Dict[str, Any]], **kwargs) -> TaskResult:
        """
        Execute a task using this agent.
        
        Args:
            task: The task to execute, either as a string command or a structured task dict
            **kwargs: Additional task-specific parameters
            
        Returns:
            A TaskResult object containing the execution results and metadata
        """
        pass
```

### Agent Types

```python
class AgentType(str, Enum):
    """Enumeration of supported agent types."""
    
    CONVERSATIONAL = "conversational"
    TASK_ORIENTED = "task_oriented"
    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"
    CUSTOM = "custom"
```

### Agent Configuration

```python
class AgentConfig(BaseModel):
    """Configuration parameters for an Agent."""
    
    max_retries: int = Field(default=3, description="Maximum number of task retry attempts")
    timeout_seconds: int = Field(default=60, description="Task timeout in seconds")
    allow_external_calls: bool = Field(default=False, description="Whether the agent can make external API calls")
    memory_size: int = Field(default=10, description="Number of previous interactions to remember")
    verbose: bool = Field(default=False, description="Enable verbose logging")
```

### Task Results

```python
class TaskResult(BaseModel):
    """Result of a task execution by an Agent."""
    
    agent_id: str = Field(..., description="ID of the agent that executed the task")
    task_id: str = Field(default_factory=lambda: str(id({})), description="Unique ID for this task")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="When the task was executed")
    success: bool = Field(..., description="Whether the task was successful")
    result: Optional[Any] = Field(default=None, description="Task execution result data")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the task execution")
```

## RAG API

### RAGPipeline Class

```python
class RAGPipeline:
    """
    A pipeline for Retrieval-Augmented Generation (RAG).
    
    The RAG pipeline connects to various knowledge sources, retrieves relevant
    information based on queries, and augments agent capabilities with this
    additional context.
    """
    
    def __init__(self, name: str = "default_pipeline"):
        """
        Initialize a new RAG pipeline.
        
        Args:
            name: Optional name for the pipeline
        """
        pass
    
    def add_source(self, source_name: str, source: Union[KnowledgeSource, str]) -> None:
        """
        Add a knowledge source to this RAG pipeline.
        
        Args:
            source_name: Name to identify this source
            source: Either a KnowledgeSource object or a string URL/path to create a source
        
        Raises:
            ValueError: If a source with the same name already exists
        """
        pass
    
    def remove_source(self, source_name: str) -> bool:
        """
        Remove a knowledge source from this RAG pipeline.
        
        Args:
            source_name: Name of the source to remove
            
        Returns:
            True if the source was removed, False if it doesn't exist
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant information from all connected knowledge sources.
        
        Args:
            query: The query to retrieve information for
            top_k: The maximum number of results to return per source
            
        Returns:
            A dictionary mapping source names to lists of retrieved documents/results
        """
        pass
    
    def combine_sources(self, *source_names: str, new_name: str) -> bool:
        """
        Combine multiple sources into a single virtual source.
        
        Args:
            *source_names: Names of sources to combine
            new_name: Name for the new combined source
            
        Returns:
            True if sources were successfully combined, False otherwise
            
        Raises:
            ValueError: If fewer than two sources are provided or new_name already exists
        """
        pass
```

### KnowledgeSource Class

```python
class KnowledgeSource(ABC):
    """
    Abstract base class for all knowledge sources.
    
    A knowledge source provides methods to search and retrieve information
    based on queries. Different implementations can connect to different
    types of data sources (APIs, databases, files, etc.).
    """
    
    def __init__(self, name: str):
        """
        Initialize a new knowledge source.
        
        Args:
            name: A name for this knowledge source
        """
        pass
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search this knowledge source for relevant information.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            A list of retrieved documents or information pieces
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        pass
```

## ML Model API

### MLModel Class

```python
class MLModel(ABC):
    """
    Base class for machine learning models in AutoAI-AgentRAG.
    
    This abstract class defines the interface for all ML models integrated
    into the framework, regardless of the underlying framework (TensorFlow,
    PyTorch, etc.).
    """
    
    def __init__(
        self, 
        name: str,
        framework: ModelFramework = ModelFramework.CUSTOM,
        config: Optional[ModelConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new ML model.
        
        Args:
            name: Name of the model
            framework: ML framework used by this model
            config: Optional configuration parameters
            metadata: Optional metadata for the model
        """
        pass
    
    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model from disk or initialize it.
        
        Args:
            model_path: Optional path to model weights or definition
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the model with the given inputs.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        pass
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'MLModel':
        """
        Create a model from a pretrained checkpoint or model repository.
        
        Args:
            model_name_or_path: Name of a pretrained model or path to model weights
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An initialized MLModel instance
        """
        pass
```

### Framework-Specific Models

#### TensorFlowModel

```python
class TensorFlowModel(MLModel):
    """
    TensorFlow-specific implementation of the MLModel class.
    
    This class provides TensorFlow-specific methods for loading, saving, and
    running inference with TensorFlow models.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[TensorFlowConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new TensorFlow model.
        
        Args:
            name: Name of the model
            config: Optional TensorFlow-specific configuration
            metadata: Optional metadata for the model
        """
        pass
    
    def convert_to_tflite(self, save_path: str, optimization: str = "DEFAULT") -> bool:
        """
        Convert the model to TensorFlow Lite format.
        
        Args:
            save_path: Path to save the TFLite model to
            optimization: Optimization level (DEFAULT, OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY)
            
        Returns:
            True if the model was converted and saved successfully, False otherwise
        """
        pass
```

#### PyTorchModel

```python
class PyTorchModel(MLModel):
    """
    PyTorch-specific implementation of the MLModel class.
    
    This class provides PyTorch-specific methods for loading, saving, and
    running inference with PyTorch models.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[PyTorchConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new PyTorch model.
        
        Args:
            name: Name of the model
            config: Optional PyTorch-specific configuration
            metadata: Optional metadata for the model
        """
        pass
```

## CLI API

The CLI API provides a command-line interface for working with AutoAI-AgentRAG.

### Main CLI Group

```python
@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """
    AutoAI-AgentRAG CLI - A tool for managing AI agents with RAG capabilities.
    
    This CLI provides commands for initializing projects, training models,
    deploying agents, and more.
    """
    pass
```

### Init Command

```python
@cli.command()
@click.argument("project_name")
@click.option(
    "--template", "-t",
    type=click.Choice(["basic", "full", "minimal"]),
    default="basic",
    help="Project template to use"
)
@click.option(
    "--directory", "-d", 
    type=click.Path(exists=False, file_okay=False), 
    default=".",
    help="Parent directory for the new project"
)
def init(project_name: str, template: str, directory: str):
    """
    Initialize a new AutoAI-AgentRAG project.
    
    Creates a new project with the given name using the specified template
    in the specified directory.
    """
    pass
```

### Train Command

```python
@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    help="Model type or path to train"
)
@click.option(
    "--data", "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data"
)
# ... additional options ...
def train(model: str, data: str, output: str, epochs: int, batch_size: int, config: Optional[str]):
    """
    Train a machine learning model.
    
    Trains a model using the specified data and parameters, and saves it to the
    specified output directory.
    """
    pass
```

### Deploy Command

```python
@cli.command()
@click.option(
    "--agent", "-a",
    required=True,
    help="Agent name or configuration file to deploy"
)
@click.option(
    "--port", "-p",
    type=int,
    default=8000,
    help="Port to run the agent server on"
)
# ... additional options ...
def deploy(agent: str, port: int, host: str, models_dir: str, config: Optional[str], rag: bool):
    """
    Deploy an AI agent with an optional web interface.
    
    This command starts a web server that hosts the specified agent,
    making it available for API calls and web dashboard monitoring.
    """
    pass
```

## Utility Functions

### Logging Utilities

```python
def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose (DEBUG) logging
    """
    pass
```

### File Utilities

```python
def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory to ensure exists
    """
    pass
```

### Model Utilities

```python
def infer_framework(model_path: str) -> ModelFramework:
    """
    Infer the ML framework based on the model file or name.
    
    Args:
        model_path: Path to model file or name of model
        
    Returns:

# API Reference

This page provides detailed documentation for the AutoAI-AgentRAG API.

## Agent API

### Agent Class

```python
class Agent:
    """
    The core Agent class for building intelligent automation agents.
    
    Agents can be configured with different capabilities, connect to RAG systems,
    and utilize ML models to perform complex tasks.
    """
    
    def __init__(
        self, 
        name: str, 
        agent_type: AgentType = AgentType.TASK_ORIENTED,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize a new Agent instance.
        
        Args:
            name: A human-readable name for the agent
            agent_type: The type of agent functionality to enable
            config: Optional configuration parameters
        """
        pass
        
    def connect_rag(self, rag_pipeline: RAGPipeline) -> None:
        """
        Connect a RAG pipeline to this agent for knowledge retrieval.
        
        Args:
            rag_pipeline: The RAG pipeline to connect to this agent
        """
        pass
    
    def add_model(self, model: MLModel, model_name: Optional[str] = None) -> None:
        """
        Add a machine learning model to this agent.
        
        Args:
            model: The machine learning model to add
            model_name: Optional custom name for the model. If not provided, 
                       the model's default name will be used.
        """
        pass
    
    def execute(self, task: Union[str, Dict[str, Any]], **kwargs) -> TaskResult:
        """
        Execute a task using this agent.
        
        Args:
            task: The task to execute, either as a string command or a structured task dict
            **kwargs: Additional task-specific parameters
            
        Returns:
            A TaskResult object containing the execution results and metadata
        """
        pass
```

### Agent Types

```python
class AgentType(str, Enum):
    """Enumeration of supported agent types."""
    
    CONVERSATIONAL = "conversational"
    TASK_ORIENTED = "task_oriented"
    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"
    CUSTOM = "custom"
```

### Agent Configuration

```python
class AgentConfig(BaseModel):
    """Configuration parameters for an Agent."""
    
    max_retries: int = Field(default=3, description="Maximum number of task retry attempts")
    timeout_seconds: int = Field(default=60, description="Task timeout in seconds")
    allow_external_calls: bool = Field(default=False, description="Whether the agent can make external API calls")
    memory_size: int = Field(default=10, description="Number of previous interactions to remember")
    verbose: bool = Field(default=False, description="Enable verbose logging")
```

### Task Results

```python
class TaskResult(BaseModel):
    """Result of a task execution by an Agent."""
    
    agent_id: str = Field(..., description="ID of the agent that executed the task")
    task_id: str = Field(default_factory=lambda: str(id({})), description="Unique ID for this task")
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="When the task was executed")
    success: bool = Field(..., description="Whether the task was successful")
    result: Optional[Any] = Field(default=None, description="Task execution result data")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the task execution")
```

## RAG API

### RAGPipeline Class

```python
class RAGPipeline:
    """
    A pipeline for Retrieval-Augmented Generation (RAG).
    
    The RAG pipeline connects to various knowledge sources, retrieves relevant
    information based on queries, and augments agent capabilities with this
    additional context.
    """
    
    def __init__(self, name: str = "default_pipeline"):
        """
        Initialize a new RAG pipeline.
        
        Args:
            name: Optional name for the pipeline
        """
        pass
    
    def add_source(self, source_name: str, source: Union[KnowledgeSource, str]) -> None:
        """
        Add a knowledge source to this RAG pipeline.
        
        Args:
            source_name: Name to identify this source
            source: Either a KnowledgeSource object or a string URL/path to create a source
        
        Raises:
            ValueError: If a source with the same name already exists
        """
        pass
    
    def remove_source(self, source_name: str) -> bool:
        """
        Remove a knowledge source from this RAG pipeline.
        
        Args:
            source_name: Name of the source to remove
            
        Returns:
            True if the source was removed, False if it doesn't exist
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant information from all connected knowledge sources.
        
        Args:
            query: The query to retrieve information for
            top_k: The maximum number of results to return per source
            
        Returns:
            A dictionary mapping source names to lists of retrieved documents/results
        """
        pass
    
    def combine_sources(self, *source_names: str, new_name: str) -> bool:
        """
        Combine multiple sources into a single virtual source.
        
        Args:
            *source_names: Names of sources to combine
            new_name: Name for the new combined source
            
        Returns:
            True if sources were successfully combined, False otherwise
            
        Raises:
            ValueError: If fewer than two sources are provided or new_name already exists
        """
        pass
```

### KnowledgeSource Class

```python
class KnowledgeSource(ABC):
    """
    Abstract base class for all knowledge sources.
    
    A knowledge source provides methods to search and retrieve information
    based on queries. Different implementations can connect to different
    types of data sources (APIs, databases, files, etc.).
    """
    
    def __init__(self, name: str):
        """
        Initialize a new knowledge source.
        
        Args:
            name: A name for this knowledge source
        """
        pass
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search this knowledge source for relevant information.
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
            
        Returns:
            A list of retrieved documents or information pieces
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: The ID of the document to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        pass
```

## ML Model API

### MLModel Class

```python
class MLModel(ABC):
    """
    Base class for machine learning models in AutoAI-AgentRAG.
    
    This abstract class defines the interface for all ML models integrated
    into the framework, regardless of the underlying framework (TensorFlow,
    PyTorch, etc.).
    """
    
    def __init__(
        self, 
        name: str,
        framework: ModelFramework = ModelFramework.CUSTOM,
        config: Optional[ModelConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new ML model.
        
        Args:
            name: Name of the model
            framework: ML framework used by this model
            config: Optional configuration parameters
            metadata: Optional metadata for the model
        """
        pass
    
    @abstractmethod
    def load(self, model_path: Optional[str] = None) -> bool:
        """
        Load the model from disk or initialize it.
        
        Args:
            model_path: Optional path to model weights or definition
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference on the model with the given inputs.
        
        Args:
            inputs: Input data for model inference
            **kwargs: Additional inference parameters
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            save_path: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        pass
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> 'MLModel':
        """
        Create a model from a pretrained checkpoint or model repository.
        
        Args:
            model_name_or_path: Name of a pretrained model or path to model weights
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An initialized MLModel instance
        """
        pass
```

### Framework-Specific Models

#### TensorFlowModel

```python
class TensorFlowModel(MLModel):
    """
    TensorFlow-specific implementation of the MLModel class.
    
    This class provides TensorFlow-specific methods for loading, saving, and
    running inference with TensorFlow models.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[TensorFlowConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new TensorFlow model.
        
        Args:
            name: Name of the model
            config: Optional TensorFlow-specific configuration
            metadata: Optional metadata for the model
        """
        pass
    
    def convert_to_tflite(self, save_path: str, optimization: str = "DEFAULT") -> bool:
        """
        Convert the model to TensorFlow Lite format.
        
        Args:
            save_path: Path to save the TFLite model to
            optimization: Optimization level (DEFAULT, OPTIMIZE_FOR_SIZE, OPTIMIZE_FOR_LATENCY)
            
        Returns:
            True if the model was converted and saved successfully, False otherwise
        """
        pass
```

#### PyTorchModel

```python
class PyTorchModel(MLModel):
    """
    PyTorch-specific implementation of the MLModel class.
    
    This class provides PyTorch-specific methods for loading, saving, and
    running inference with PyTorch models.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[PyTorchConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new PyTorch model.
        
        Args:
            name: Name of the model
            config: Optional PyTorch-specific configuration
            metadata: Optional metadata for the model
        """
        pass
```

## CLI API

The CLI API provides a command-line interface for working with AutoAI-AgentRAG.

### Main CLI Group

```python
@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool):
    """
    AutoAI-AgentRAG CLI - A tool for managing AI agents with RAG capabilities.
    
    This CLI provides commands for initializing projects, training models,
    deploying agents, and more.
    """
    pass
```

### Init Command

```python
@cli.command()
@click.argument("project_name")
@click.option(
    "--template", "-t",
    type=click.Choice(["basic", "full", "minimal"]),
    default="basic",
    help="Project template to use"
)
@click.option(
    "--directory", "-d", 
    type=click.Path(exists=False, file_okay=False), 
    default=".",
    help="Parent directory for the new project"
)
def init(project_name: str, template: str, directory: str):
    """
    Initialize a new AutoAI-AgentRAG project.
    
    Creates a new project with the given name using the specified template
    in the specified directory.
    """
    pass
```

### Train Command

```python
@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    help="Model type or path to train"
)
@click.option(
    "--data", "-d",
    required=True,
    type=click.Path(exists=True),
    help="Path to training data"
)
# ... additional options ...
def train(model: str, data: str, output: str, epochs: int, batch_size: int, config: Optional[str]):
    """
    Train a machine learning model.
    
    Trains a model using the specified data and parameters, and saves it to the
    specified output directory.
    """
    pass
```

### Deploy Command

```python
@cli.command()
@click.option(
    "--agent", "-a",
    required=True,
    help="Agent name or configuration file to deploy"
)
@click.option(
    "--port", "-p",
    type=int,
    default=8000,
    help="Port to run the agent server on"
)
# ... additional options ...
def deploy(agent: str, port: int, host: str, models_dir: str, config: Optional[str], rag: bool):
    """
    Deploy an AI agent with an optional web interface.
    
    This command starts a web server that hosts the specified agent,
    making it available for API calls and web dashboard monitoring.
    """
    pass
```

## Utility Functions

### Logging Utilities

```python
def setup_logging(verbose: bool = False):
    """
    Set up logging configuration.
    
    Args:
        verbose: Whether to enable verbose (DEBUG) logging
    """
    pass
```

### File Utilities

```python
def ensure_directory(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory to ensure exists
    """
    pass
```

### Model Utilities

```python
def infer_framework(model_path: str) -> ModelFramework:
    """
    Infer the ML framework based on the model file or name.
    
    Args:
        model_path: Path to model file or name of model
        
    Returns:
        The inferred ModelFramework
    """
    pass
```

