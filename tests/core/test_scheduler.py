"""Tests for the TaskScheduler."""

import pytest
import anyio
import time
import logging
from unittest.mock import Mock, call # If needed later for finer-grained mocking

from src.core.agents.base import BaseAgent
from src.core.scheduler import TaskScheduler

# Mark all tests in this module to be run by anyio
pytestmark = pytest.mark.anyio

# Configure logging for tests (optional, can help debugging)
log = logging.getLogger(__name__)

# --- Mock Agent for Testing ---
class MockAgent(BaseAgent):
    """A mock agent for testing the scheduler."""
    def __init__(self, agent_id: str, delay: float = 0.1, fail: bool = False, run_callback: Mock = None):
        self.agent_id = agent_id
        self.delay = delay
        self.fail = fail
        self.started = False
        self.finished = False
        self.start_time = 0.0
        self.end_time = 0.0
        self.run_callback = run_callback # Optional mock to track calls

    async def run(self) -> None:
        """Simulates agent execution with optional delay and failure."""
        log.info(f"MockAgent {self.agent_id}: Starting run (delay={self.delay}s, fail={self.fail})")
        self.started = True
        self.start_time = time.monotonic()
        if self.run_callback:
            self.run_callback(self.agent_id, 'start')

        await anyio.sleep(self.delay)

        if self.fail:
            self.end_time = time.monotonic()
            if self.run_callback:
                self.run_callback(self.agent_id, 'fail')
            log.warning(f"MockAgent {self.agent_id}: Simulating failure.")
            raise RuntimeError(f"Agent {self.agent_id} failed as requested.")

        self.finished = True
        self.end_time = time.monotonic()
        if self.run_callback:
            self.run_callback(self.agent_id, 'finish')
        log.info(f"MockAgent {self.agent_id}: Finished run.")

# --- Test Cases --- 

async def test_scheduler_initialization():
    """Test TaskScheduler initialization."""
    scheduler = TaskScheduler(max_concurrent_tasks=3)
    assert scheduler.max_concurrent_tasks == 3
    assert scheduler._tasks == []

    with pytest.raises(ValueError):
        TaskScheduler(max_concurrent_tasks=0)
    with pytest.raises(ValueError):
        TaskScheduler(max_concurrent_tasks=-1)

async def test_add_task():
    """Test adding valid and invalid tasks."""
    scheduler = TaskScheduler()
    agent1 = MockAgent(agent_id="agent1")
    agent2 = MockAgent(agent_id="agent2")

    scheduler.add_task(agent1)
    scheduler.add_task(agent2)
    assert len(scheduler._tasks) == 2
    assert scheduler._tasks[0] is agent1
    assert scheduler._tasks[1] is agent2

    with pytest.raises(TypeError):
        scheduler.add_task("not an agent") # type: ignore

async def test_run_basic_execution():
    """Test that the scheduler runs added tasks successfully."""
    scheduler = TaskScheduler(max_concurrent_tasks=2)
    agent1 = MockAgent(agent_id="agent1", delay=0.1)
    agent2 = MockAgent(agent_id="agent2", delay=0.1)

    scheduler.add_task(agent1)
    scheduler.add_task(agent2)

    await scheduler.run()

    assert agent1.started
    assert agent1.finished
    assert not agent1.fail # Ensure it didn't fail
    assert agent2.started
    assert agent2.finished
    assert not agent2.fail

async def test_concurrency_limit():
    """Test that the scheduler respects the concurrency limit."""
    max_concurrent = 2
    num_agents = 4
    agent_delay = 0.2 # Short delay to observe overlap
    total_expected_time_min = (num_agents / max_concurrent) * agent_delay # Rough lower bound

    scheduler = TaskScheduler(max_concurrent_tasks=max_concurrent)
    callback_mock = Mock()
    agents = [MockAgent(agent_id=f"agent{i}", delay=agent_delay, run_callback=callback_mock) for i in range(num_agents)]

    for agent in agents:
        scheduler.add_task(agent)

    start_time = time.monotonic()
    await scheduler.run()
    end_time = time.monotonic()

    # Check all agents completed
    for agent in agents:
        assert agent.started
        assert agent.finished

    # Check timing - should take roughly two 'batches' of agents
    assert (end_time - start_time) >= total_expected_time_min
    # Add a buffer for scheduling overhead
    assert (end_time - start_time) < total_expected_time_min + agent_delay + 0.1 

    # Use the callback mock to verify concurrency
    max_observed_concurrent = 0
    current_concurrent = 0
    start_times = {}
    finish_times = {}

    for call_item in callback_mock.call_args_list:
        agent_id, event_type = call_item.args
        timestamp = time.monotonic() # Use call time approx.

        if event_type == 'start':
            current_concurrent += 1
            start_times[agent_id] = timestamp
            max_observed_concurrent = max(max_observed_concurrent, current_concurrent)
        elif event_type == 'finish' or event_type == 'fail':
            current_concurrent -= 1
            finish_times[agent_id] = timestamp
    
    assert max_observed_concurrent <= max_concurrent
    # Make sure the counter went back to 0
    assert current_concurrent == 0 

async def test_error_handling_one_agent_fails():
    """Test that the scheduler continues if one agent fails."""
    scheduler = TaskScheduler(max_concurrent_tasks=2)
    agent_ok1 = MockAgent(agent_id="ok1", delay=0.1)
    agent_fail = MockAgent(agent_id="fail1", delay=0.1, fail=True)
    agent_ok2 = MockAgent(agent_id="ok2", delay=0.1)

    scheduler.add_task(agent_ok1)
    scheduler.add_task(agent_fail)
    scheduler.add_task(agent_ok2)

    # We expect the scheduler's run method to complete without raising the agent's error
    await scheduler.run()

    # Check that the successful agents completed
    assert agent_ok1.started
    assert agent_ok1.finished
    assert not agent_ok1.fail
    assert agent_ok2.started
    assert agent_ok2.finished
    assert not agent_ok2.fail

    # Check that the failing agent started but did not finish
    assert agent_fail.started
    assert not agent_fail.finished
    assert agent_fail.fail

# TODO: Add tests for error handling
