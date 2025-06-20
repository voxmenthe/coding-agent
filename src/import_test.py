import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Configure basic logging for the test

try:
    from google import genai
    from google.genai import types
    print("Successfully imported google.genai and google.genai.types")
    logger.info("Successfully imported google.genai and google.genai.types")
except ImportError as e:
    print(f"Failed to import google.genai: {e}")
    logger.error(f"Failed to import google.genai: {e}", exc_info=True)
    raise
