import anyio
import logging
from typing import List

from .agents.base import BaseAgent

# Configure logging for the scheduler
log = logging.getLogger(__name__)
# Use a standard format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TaskScheduler:
    """Manages the concurrent execution of agent tasks using AnyIO."""

    def __init__(self, max_concurrent_tasks: int = 5):
        """Initializes the TaskScheduler.

        Args:
            max_concurrent_tasks: The maximum number of agent tasks to run concurrently.
        """
        if max_concurrent_tasks < 1:
            raise ValueError("max_concurrent_tasks must be at least 1")
        self.max_concurrent_tasks = max_concurrent_tasks
        self._tasks: List[BaseAgent] = [] # Store agent instances to run
        log.info(f"TaskScheduler initialized with max_concurrent_tasks={self.max_concurrent_tasks}")

    def add_task(self, agent_instance: BaseAgent) -> None:
        """Adds an agent instance to the list of tasks to be run."""
        if not isinstance(agent_instance, BaseAgent):
            raise TypeError("agent_instance must be an instance of BaseAgent or its subclass")
        self._tasks.append(agent_instance)
        # Attempt to get agent_id for logging, default to class name if not present
        agent_id = getattr(agent_instance, 'agent_id', 'N/A')
        log.info(f"Added task for agent: {agent_instance.__class__.__name__} (ID: {agent_id})")

    # --- run, _run_internal, _run_agent_task methods will be added next ---

    async def run(self) -> None:
        """Runs all added tasks concurrently, respecting the concurrency limit."""
        log.info(f"Starting scheduler run with {len(self._tasks)} tasks.")
        if not self._tasks:
            log.warning("No tasks added to the scheduler. Exiting run.")
            return

        try:
            async with anyio.create_task_group() as tg:
                semaphore = anyio.Semaphore(self.max_concurrent_tasks)
                for agent_instance in self._tasks:
                    tg.start_soon(self._run_agent_task, agent_instance, semaphore)
            log.info("Scheduler run finished successfully.")
        except Exception as e:
            log.exception("An error occurred during scheduler run.")
            # Depending on desired behavior, could re-raise or handle

    async def _run_agent_task(self, agent_instance: BaseAgent, semaphore: anyio.Semaphore) -> None:
        """Internal method to run a single agent task, managing the semaphore and errors."""
        agent_id = getattr(agent_instance, 'agent_id', 'N/A')
        agent_name = agent_instance.__class__.__name__
        async with semaphore:
            log.info(f"Starting execution for agent: {agent_name} (ID: {agent_id})")
            try:
                # Assuming BaseAgent.run is async as defined
                await agent_instance.run()
                log.info(f"Finished execution for agent: {agent_name} (ID: {agent_id})")
            except Exception as e:
                log.exception(f"Error executing agent: {agent_name} (ID: {agent_id})")
                # Decide how to handle agent errors (e.g., stop scheduler? continue?)
