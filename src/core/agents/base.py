from abc import ABC, abstractmethod
import logging
import abc

# Configure logging for the base agent module
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class BaseAgent(abc.ABC):
    """Abstract base class for all agents."""

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

    @abc.abstractmethod
    async def run(self) -> None:
        """The main execution logic for the agent."""
        raise NotImplementedError

    # Optional: Add common helper methods here if needed by multiple agents
    # For example, interacting with a shared configuration or state manager.

# Example of a concrete agent inheriting from BaseAgent (for illustration)
# class MyConcreteAgent(BaseAgent):
#     def __init__(self, agent_id: str, specific_param: str, **kwargs):
#         super().__init__(agent_id=agent_id, **kwargs)
#         self.specific_param = specific_param
#         log.info(f"MyConcreteAgent specific param: {self.specific_param}")

#     async def run(self) -> None:
#         log.info(f"[{self.name} / {self.agent_id}] Running step.")
#         # --- Agent logic here ---
#         log.info(f"[{self.name} / {self.agent_id}] Step finished.")
