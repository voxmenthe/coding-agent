# examples/core/example_scheduler_usage.py

import anyio
import time
import random
import logging
from typing import Optional

# Assuming your project structure allows this import
# Adjust the path based on your actual project structure and how you run examples
# (e.g., running from the root directory with python -m examples.core.example...)
try:
    from src.core.scheduler import TaskScheduler
    from src.core.agents.base import BaseAgent
except ImportError:
    # Fallback for running the script directly from the examples/core directory
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.core.scheduler import TaskScheduler
    from src.core.agents.base import BaseAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Mock Agent for Demonstration ---

class MockAgent(BaseAgent):
    """A simple mock agent that simulates work with a delay."""
    _instances_running = 0
    _lock = anyio.Lock()

    def __init__(self, agent_id: str, work_duration: float = 1.0, fail_rate: float = 0.0):
        self.agent_id = agent_id
        self.work_duration = work_duration
        self.fail_rate = fail_rate
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.succeeded: Optional[bool] = None

    async def run(self):
        self.start_time = time.monotonic()
        log.info(f"Agent [{self.agent_id}] starting work (duration: {self.work_duration}s).")

        async with self._lock:
            MockAgent._instances_running += 1
            log.info(f"Agent [{self.agent_id}] acquired lock. Instances running: {MockAgent._instances_running}")

        try:
            # Simulate async work
            await anyio.sleep(self.work_duration)

            # Simulate potential failure
            if random.random() < self.fail_rate:
                self.succeeded = False
                log.error(f"Agent [{self.agent_id}] encountered a simulated failure.")
                raise ValueError(f"Simulated failure in agent {self.agent_id}")
            else:
                self.succeeded = True
                log.info(f"Agent [{self.agent_id}] completed work successfully.")

        finally:
            async with self._lock:
                MockAgent._instances_running -= 1
                log.info(f"Agent [{self.agent_id}] released lock. Instances running: {MockAgent._instances_running}")
            self.end_time = time.monotonic()

    def __str__(self):
        status = "Not Run"
        duration_str = ""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            duration_str = f", Duration: {duration:.2f}s"
            if self.succeeded is True:
                status = "Success"
            elif self.succeeded is False:
                status = "Failed"
            else: # Should not happen if run completed
                status = "Ended (Unknown Status)"
        elif self.start_time:
            status = "Running"

        return f"Agent ID: {self.agent_id}, Status: {status}{duration_str}"

# --- Main Example Function ---

async def run_scheduler_example():
    """Demonstrates the usage of the TaskScheduler."""
    log.info("--- TaskScheduler Example ---")

    max_concurrent = 3
    num_agents = 7
    base_work_duration = 0.5
    fail_rate = 0.1 # 10% chance of failure per agent

    log.info(f"Initializing TaskScheduler with max_concurrent_tasks = {max_concurrent}")
    scheduler = TaskScheduler(max_concurrent_tasks=max_concurrent)

    log.info(f"Creating {num_agents} MockAgent instances...")
    agents = []
    for i in range(num_agents):
        # Vary work duration slightly
        duration = base_work_duration + random.uniform(0, 1.0)
        agent = MockAgent(agent_id=f"Agent_{i+1:02d}", work_duration=duration, fail_rate=fail_rate)
        agents.append(agent)
        scheduler.add_task(agent)
        log.info(f"Added task: {agent.agent_id} (duration ~{duration:.2f}s)")

    log.info("--- Running Scheduler ---")
    start_run_time = time.monotonic()
    await scheduler.run()
    end_run_time = time.monotonic()
    log.info("--- Scheduler Run Complete ---")

    log.info(f"Total scheduler run time: {end_run_time - start_run_time:.2f}s")
    log.info("--- Agent Results ---")
    successful_agents = 0
    failed_agents = 0
    for agent in agents:
        log.info(str(agent))
        if agent.succeeded is True:
            successful_agents += 1
        elif agent.succeeded is False:
            failed_agents += 1

    log.info(f"Summary: {successful_agents} successful, {failed_agents} failed.")
    log.info("--- End of Example ---")


if __name__ == "__main__":
    try:
        anyio.run(run_scheduler_example)
    except KeyboardInterrupt:
        log.info("Scheduler example interrupted by user.")
