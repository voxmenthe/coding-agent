import asyncio
import threading
import uuid
import logging
from typing import Optional, Callable, Dict, Any, List, Coroutine

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)

class AsyncTaskManager:
    def __init__(self, main_app_handler: Optional[Any] = None):
        self.main_app_handler = main_app_handler
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AsyncTaskManagerThread")
        self.active_background_tasks: Dict[str, Dict[str, Any]] = {}
        # self.progress_bars: Dict[str, Progress] = {} # Store Progress objects per task
        self._thread.start()
        logger.info("AsyncTaskManager initialized and event loop thread started.")

    def _run_loop(self):
        logger.info("Async task manager event loop started.")
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
            logger.info("Async task manager event loop closed.")

    def shutdown(self):
        logger.info("AsyncTaskManager shutdown requested.")
        if self.loop.is_running():
            logger.info("Stopping asyncio event loop.")
            self.loop.call_soon_threadsafe(self.loop.stop)

        # Wait for the thread to finish, with a timeout
        # Ensure this is called from the main thread, not the loop's thread.
        if threading.current_thread() != self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("AsyncTaskManager thread did not shut down gracefully.")
        logger.info("AsyncTaskManager shutdown complete.")

    def _on_task_done(self, task_id: str, task_name: str, progress_bar: Progress, rich_task_id: Any, future: asyncio.Future):
        """Callback executed when a background task finishes."""
        try:
            result = future.result()  # Raise exception if task failed
            logger.info(f"Task '{task_name}' (ID: {task_id}) completed successfully. Result: {result}")
            # For display, we might want to truncate long results if printed directly here.
            # For now, assuming main_app_handler will handle detailed result display.
            if self.main_app_handler and hasattr(self.main_app_handler, 'update_task_status_display'):
                self.main_app_handler.update_task_status_display(task_id, f"✅ {task_name} completed.")
            else:
                print(f"\n✅ Task '{task_name}' (ID: {task_id}) completed successfully.")


            task_info = self.active_background_tasks.get(task_id, {})
            meta = task_info.get("meta", {})

            if meta.get("type") == "script_execution":
                if self.main_app_handler and hasattr(self.main_app_handler, 'handle_script_completion'):
                    self.main_app_handler.handle_script_completion(task_id, task_name, str(result))
                else:
                    logger.warning(f"No main_app_handler or handle_script_completion method to process script output for task {task_id}.")
            elif meta.get("type") == "pdf_processing": # Example for PDF processing
                if self.main_app_handler and hasattr(self.main_app_handler, 'handle_pdf_completion'):
                    # Result here might be the GenAI file object or a status message
                    self.main_app_handler.handle_pdf_completion(task_id, task_name, result)
                else:
                    logger.warning(f"No main_app_handler or handle_pdf_completion method for task {task_id}.")


        except asyncio.CancelledError:
            logger.warning(f"Task '{task_name}' (ID: {task_id}) was cancelled.")
            if self.main_app_handler and hasattr(self.main_app_handler, 'update_task_status_display'):
                self.main_app_handler.update_task_status_display(task_id, f"🚫 {task_name} cancelled.")
            else:
                print(f"\n🚫 Task '{task_name}' (ID: {task_id}) was cancelled.")
        except Exception as e:
            logger.error(f"Task '{task_name}' (ID: {task_id}) failed.", exc_info=True)
            if self.main_app_handler and hasattr(self.main_app_handler, 'update_task_status_display'):
                 self.main_app_handler.update_task_status_display(task_id, f"❌ {task_name} error: {type(e).__name__}")
            else:
                print(f"\n❌ Task '{task_name}' (ID: {task_id}) failed: {type(e).__name__}: {e}")
        finally:
            self.active_background_tasks.pop(task_id, None)
            # Stop the specific Rich progress task, not the whole Progress object if it's shared.
            # If each task has its own Progress object, then stop it.
            # For now, assuming progress_bar is the Rich Progress object itself, and rich_task_id is the TaskID from progress.add_task
            if progress_bar and rich_task_id is not None:
                 # This logic might need refinement based on how Progress is used.
                 # If a single Progress object is used for all tasks, we update, not stop.
                 # If each task has its own Progress object, then stop it.
                 # The original code created a new Progress object per task.
                 progress_bar.update(rich_task_id, completed=progress_bar.tasks[0].total if progress_bar.tasks else 100) # Mark as complete
                 progress_bar.stop() # Stop this progress instance.

            # Refresh prompt if main_app_handler supports it
            if self.main_app_handler and hasattr(self.main_app_handler, 'refresh_prompt_display'):
                self.main_app_handler.refresh_prompt_display()


    def submit_task(self, coro_creator: Callable[..., Coroutine[Any, Any, Any]],
                    task_name: str, progress_total: float = 100.0,
                    task_meta: Optional[Dict[str, Any]] = None) -> str:
        task_id = str(uuid.uuid4())

        # Each task gets its own Progress display instance for now
        # This might be noisy if many tasks run; consider a shared Progress object if main_app_handler can manage it.
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            # transient=True # Keep progress visible until explicitly cleared or handled by UI
        )
        rich_task_id = progress.add_task(description=f"Initializing {task_name}...", total=progress_total, start=False)

        # The coroutine created by coro_creator must accept these arguments:
        # task_id, progress_bar (the Rich Progress object), rich_task_id (the ID for progress.update)
        # plus any specific arguments it needs, passed via functools.partial when calling submit_task.
        task_coro = coro_creator(task_id=task_id, progress_bar=progress, rich_task_id=rich_task_id)

        # Start the Rich Progress display for this task
        # This needs to be handled carefully if run from a non-main thread or if prompt_toolkit is active.
        # For now, let's assume direct printing of progress might interfere with prompt_toolkit.
        # A better approach is for main_app_handler to manage Rich display.
        # For simplicity here, we'll let Rich print, but this is a known issue for CLI apps.
        # TODO: Integrate Rich Progress display with prompt_toolkit UI if possible, or use a simpler non-Rich progress.
        # For now, we pass `progress` and `rich_task_id` to the coroutine, which can update it.
        # The actual `progress.start()` or `Live` context should be managed by the UI layer ideally.
        # However, since the original code started progress here, we'll keep a simplified version.
        # This part is tricky with prompt_toolkit. For now, we'll just store it.
        # The task itself will call progress.update() and progress.start() if needed by its logic.

        fut = asyncio.run_coroutine_threadsafe(task_coro, self.loop)

        self.active_background_tasks[task_id] = {
            "future": fut,
            "name": task_name,
            "progress_bar": progress,
            "rich_task_id": rich_task_id,
            "meta": task_meta if task_meta else {}
        }

        # Add done callback (needs to be partial as it takes more than just the future)
        callback = functools.partial(self._on_task_done, task_id, task_name, progress, rich_task_id)
        fut.add_done_callback(callback)

        # Print to console (or use main_app_handler to display this)
        if self.main_app_handler and hasattr(self.main_app_handler, 'display_message'):
            self.main_app_handler.display_message(f"⏳ Task '{task_name}' (ID: {task_id}) started in background.")
        else:
            print(f"⏳ Task '{task_name}' (ID: {task_id}) started in background – you can keep chatting.")

        return task_id

    def cancel_task(self, task_id_str: str):
        task_info = self.active_background_tasks.get(task_id_str)
        if not task_info:
            message = f"\n❌ Task ID '{task_id_str}' not found or already completed."
            if self.main_app_handler and hasattr(self.main_app_handler, 'display_message'):
                self.main_app_handler.display_message(message)
            else:
                print(message)
            return

        future = task_info.get("future")
        task_name = task_info.get("name", "Unnamed Task")

        if future and not future.done():
            # The cancellation itself is thread-safe.
            cancelled = self.loop.call_soon_threadsafe(future.cancel)
            # Note: future.cancel() might return False if already done/cancelling.
            # The callback _on_task_done will handle logging and cleanup.
            message = f"\n➡️ Cancellation request sent for task '{task_name}' (ID: {task_id_str})."
        elif future and future.done():
            message = f"\nℹ️ Task '{task_name}' (ID: {task_id_str}) has already completed."
        else:
            message = f"\n⚠️ Could not cancel task '{task_name}' (ID: {task_id_str}). Future object missing or invalid state."

        if self.main_app_handler and hasattr(self.main_app_handler, 'display_message'):
            self.main_app_handler.display_message(message)
        else:
            print(message)

    def list_tasks(self):
        if not self.active_background_tasks:
            message = "\nℹ️ No active background tasks."
            if self.main_app_handler and hasattr(self.main_app_handler, 'display_message'):
                self.main_app_handler.display_message(message)
            else:
                print(message)
            return

        output_lines = ["\n📋 Active Background Tasks:"]
        for task_id, info in self.active_background_tasks.items():
            future = info.get("future")
            name = info.get("name", "Unnamed Task")
            status = "Running"
            if future:
                if future.cancelled(): status = "Cancelling"
                elif future.done(): status = "Completed (Pending Removal)"
            output_lines.append(f"  - ID: {task_id}, Name: {name}, Status: {status}")

        full_message = "\n".join(output_lines)
        if self.main_app_handler and hasattr(self.main_app_handler, 'display_message'):
            self.main_app_handler.display_message(full_message)
        else:
            print(full_message)

    def get_loop(self):
        """Returns the asyncio event loop used by the manager."""
        return self.loop

    def get_active_tasks_info(self) -> Dict[str, Dict[str, Any]]:
        """Returns a copy of the active background tasks information."""
        return dict(self.active_background_tasks)

# Example usage (for testing within the file if needed)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    class MockMainAppHandler:
        def handle_script_completion(self, task_id, task_name, script_output):
            logger.info(f"[MockMainAppHandler] Script completed: {task_id} - {task_name}. Output:\n{script_output}")

        def update_task_status_display(self, task_id, message):
            logger.info(f"[MockMainAppHandler] Task status update: {task_id} - {message}")

        def display_message(self, message):
            print(message)

        def refresh_prompt_display(self):
            logger.info("[MockMainAppHandler] Refreshing prompt display (e.g., for prompt_toolkit).")


    async def sample_long_task(task_id: str, progress_bar: Progress, rich_task_id: Any, duration: int, message: str):
        logger.info(f"Task {task_id} ({message}) started. Will run for {duration}s. RichTaskID: {rich_task_id}")
        if progress_bar: # Start the progress bar if it's passed and valid
            progress_bar.start()
            progress_bar.update(rich_task_id, description=f"Running {message}...", start=True)

        for i in range(duration):
            await asyncio.sleep(1)
            if progress_bar:
                progress_bar.update(rich_task_id, advance=100/duration, description=f"{message}: {i+1}/{duration}s")
            logger.debug(f"Task {task_id} ({message}) progress: {i+1}/{duration}")
            if i == duration // 2 and message == "Failing Task": # Simulate failure
                raise ValueError("Simulated failure in long task")
        logger.info(f"Task {task_id} ({message}) finished.")
        return f"Result from {task_id}: {message} completed after {duration}s"

    mock_handler = MockMainAppHandler()
    manager = AsyncTaskManager(main_app_handler=mock_handler)

    try:
        logger.info("Submitting tasks...")
        # Use functools.partial to prepare coro_creator with specific args for the task
        task1_coro_creator = functools.partial(sample_long_task, duration=5, message="Task 1 (Success)")
        task_id1 = manager.submit_task(task1_coro_creator, "SuccessTask1", task_meta={"type": "generic"})

        task2_coro_creator = functools.partial(sample_long_task, duration=3, message="Task 2 (Script)")
        task_id2 = manager.submit_task(task2_coro_creator, "ScriptTask2", task_meta={"type": "script_execution"})

        task3_coro_creator = functools.partial(sample_long_task, duration=6, message="Failing Task")
        task_id3 = manager.submit_task(task3_coro_creator, "FailingTask3", task_meta={"type": "generic"})

        logger.info(f"Tasks submitted: {task_id1}, {task_id2}, {task_id3}")
        manager.list_tasks()

        time.sleep(2)
        logger.info(f"Attempting to cancel Task 1 ({task_id1})")
        manager.cancel_task(task_id1)

        manager.list_tasks()

        # Keep main thread alive to see task completions
        logger.info("Main thread sleeping for 10 seconds to allow tasks to run...")
        time.sleep(10)

        manager.list_tasks()
        logger.info("Main thread finished sleeping.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Shutting down AsyncTaskManager...")
        manager.shutdown()
        logger.info("AsyncTaskManager shutdown complete.")
        logger.info("Exiting test script.")
