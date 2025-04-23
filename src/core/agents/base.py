from abc import ABC, abstractmethod
import logging

# Configure logging for the base agent module
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract Base Class for all agents in the system."""

    def __init__(self, agent_id: str, **kwargs):
        """Initializes the base agent.
        
        Args:
            agent_id: A unique identifier for this agent instance.
            **kwargs: Additional configuration parameters for the agent.
        """
        self.agent_id = agent_id
        self.config = kwargs # Store any extra config
        log.info(f"Initializing {self.__class__.__name__} with ID: {self.agent_id}")

    @property
    def name(self) -> str:
        """Returns the name of the agent class."""
        return self.__class__.__name__

    @abstractmethod
    def run_step(self, *args, **kwargs) -> any:
        """Executes a single step of the agent's logic.
        
        This method should contain the core processing for the agent.
        It will be called by the TaskScheduler, likely within a thread.

        Args:
            *args: Positional arguments specific to the agent's task.
            **kwargs: Keyword arguments specific to the agent's task.
            
        Returns:
            Any result produced by the agent's step (optional).
        """
        pass # Must be implemented by subclasses

    # Optional: Add common helper methods here if needed by multiple agents
    # For example, interacting with a shared configuration or state manager.

# Example of a concrete agent inheriting from BaseAgent (for illustration)
# class MyConcreteAgent(BaseAgent):
#     def __init__(self, agent_id: str, specific_param: str, **kwargs):
#         super().__init__(agent_id=agent_id, **kwargs)
#         self.specific_param = specific_param
#         log.info(f"MyConcreteAgent specific param: {self.specific_param}")

#     def run_step(self, input_data: str) -> str:
#         log.info(f"[{self.name} / {self.agent_id}] Running step with input: {input_data}")
#         # --- Agent logic here ---
#         result = f"Processed '{input_data}' using '{self.specific_param}'"
#         log.info(f"[{self.name} / {self.agent_id}] Step finished.")
#         return result
