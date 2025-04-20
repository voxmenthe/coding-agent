
import os
import logging

logger = logging.getLogger("Config")

# Default model if not set in environment
DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
DEFAULT_THINKING_BUDGET = 256 # Default budget (if used later)

class Config:
    """Loads and provides access to configuration settings."""

    def __init__(self):
        # Load API key from environment variable
        self.api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            # Only warn if relying on client's env var pickup
            logger.warning("⚠️ GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set.")
            logger.warning("   Agent will rely on google.genai.Client() finding it.")
            # Set to None so checks in main.py know it wasn't found *here*
            self.api_key = None

        # Load model name from environment variable, fallback to default
        self.model_name = os.environ.get('GEMINI_MODEL_NAME', DEFAULT_MODEL_NAME)
        logger.info(f"Using model: {self.model_name}")

        # Verbosity (can be overridden by command-line args in main)
        self.verbose = os.environ.get('CODE_AGENT_VERBOSE', 'False').lower() == 'true'

        # Thinking budget (can be overridden later)
        self.thinking_budget = int(os.environ.get('CODE_AGENT_BUDGET', str(DEFAULT_THINKING_BUDGET)))

    def get_api_key(self) -> str | None:
        """Returns the API key loaded from the environment."""
        return self.api_key

    def get_model_name(self, strip_prefix=False) -> str:
        """Returns the model name. Optionally strips 'models/' prefix."""
        if strip_prefix and self.model_name.startswith('models/'):
            return self.model_name.split('/', 1)[1]
        return self.model_name

    def set_model_name(self, model_name: str):
        """Allows overriding the model name (e.g., from command-line args)."""
        logger.info(f"Model name overridden to: {model_name}")
        self.model_name = model_name

    def is_verbose(self) -> bool:
        """Returns the verbosity setting."""
        return self.verbose

    def get_default_thinking_budget(self) -> int:
        """Returns the thinking budget."""
        return self.thinking_budget
