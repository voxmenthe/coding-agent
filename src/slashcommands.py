from typing import TYPE_CHECKING, List, Optional, Dict, Any
from pathlib import Path
import json
import datetime
import logging
from google.genai import types # Assuming types will be needed
from prompt_toolkit import PromptSession # Add this import
from prompt_toolkit.document import Document # Add this import

if TYPE_CHECKING:
    from .main import CodeAgent # For type hinting the agent parameter

logger = logging.getLogger(__name__)

# --- Command Handler Functions ---

def handle_help_command(agent: 'CodeAgent'):
    """Prints the help message with all available commands."""
    print("\nAvailable commands:")
    print("  /exit, /q              - Quit the agent.")
    print("  /pdf <filename> [id]   - Process a PDF file from PDFS_TO_CHAT_WITH_DIRECTORY.")
    print("  /prompt <name>         - Load a system prompt to prepend to the next message.")
    print("  /reset                 - Clear chat history and start fresh.")
    print("  /clear <n_tokens>      - Remove <tokens> from the start of history.")
    print("  /save [filename]       - Save the current conversation.")
    print("  /load <filename>       - Load a saved conversation.")
    print(f"  /thinking_budget <val> - Set tool thinking budget (current: {agent.thinking_budget}).")
    print("  /tasks                 - List active background tasks.")
    print("  /cancel <task_id>      - Attempt to cancel a background task.")
    print("  /run_script <py|sh> <path> [args...] - Run a script.")
    print("  /history <mode> <n_tokens> - Display parts of the conversation history.")
    print("  /toggle_verbose        - Toggle verbose logging on/off.")

    # Potentially list dynamically discovered prompts or other contextual help.

def handle_pdf_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /pdf command to process a PDF file."""
    # This will call agent._handle_pdf_command internally.
    # The original _handle_pdf_command in CodeAgent will remain,
    # this is just the entry point from the command parser.
    agent._handle_pdf_command(args)

def handle_prompt_command(agent: 'CodeAgent', session: PromptSession, args: List[str]):
    """Handles the /prompt command to load a prompt into the input buffer for editing."""
    if not args:
        print("\n⚠️ Usage: /prompt <prompt_name>")
        return
    prompt_name = args[0]
    prompt_content = agent._load_prompt(prompt_name) # _load_prompt is a method of CodeAgent
    if prompt_content:
        agent.prefill_prompt_content = prompt_content
        print(f"\n✅ Prompt '{prompt_name}' loaded. It will appear in your next input prompt.")
    else:
        print(f"\n❌ Prompt '{prompt_name}' not found.")

def handle_save_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /save command to save the conversation."""
    filename_arg = args[0] if args else None
    now_str = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    
    if filename_arg:
        filename = filename_arg
        if not filename.endswith('.json'):
            filename += '.json'
    else:
        filename = f'{now_str}.json'

    # Ensure SAVED_CONVERSATIONS_DIRECTORY is a Path object
    saved_conversations_dir = agent.config.get('SAVED_CONVERSATIONS_DIRECTORY')
    if isinstance(saved_conversations_dir, str):
        # This case should ideally be handled by load_config, but as a fallback:
        logger.warning("SAVED_CONVERSATIONS_DIRECTORY was a string, attempting to resolve relative to project root.")
        project_root = Path(__file__).resolve().parent.parent # slashcommands.py is in src/
        save_path_base = project_root / saved_conversations_dir
    elif isinstance(saved_conversations_dir, Path):
        save_path_base = saved_conversations_dir
    else:
        # Fallback to a default if not configured or invalid type
        logger.warning("SAVED_CONVERSATIONS_DIRECTORY not configured or invalid type, using default.")
        project_root = Path(__file__).resolve().parent.parent
        save_path_base = project_root / 'SAVED_CONVERSATIONS'
    
    save_path_base.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    save_path = save_path_base / filename

    save_state = {
        'conversation_history': [],
        'current_token_count': agent.current_token_count,
        'prompt_time_counts': agent.prompt_time_counts,
        'messages_per_interval': agent.messages_per_interval,
        '_messages_this_interval': agent._messages_this_interval,
        'active_files': [getattr(f, 'name', str(f)) for f in agent.active_files], # Ensure files are serializable
        'thinking_budget': agent.thinking_budget,
        'pending_pdf_context': agent.pending_pdf_context,
        'model_name': agent.model_name, # Save the model name used for this conversation
        'saved_at': now_str # Add a timestamp for when it was saved
    }
    serializable_history = []
    for content in agent.conversation_history: # Access via agent instance
        if hasattr(content, 'parts') and hasattr(content, 'role'):
            parts_text = [part.text for part in content.parts if hasattr(part, 'text') and part.text is not None]
            serializable_history.append({
                'role': content.role,
                'parts': parts_text
            })
        else:
            logger.warning(f"Skipping non-standard content object during save: {content}")

    save_state['conversation_history'] = serializable_history
    try:
        with open(save_path, 'w') as f:
            json.dump(save_state, f, indent=2)
        print(f"\n💾 Conversation saved as: {save_path.name} in {save_path.parent}")
    except Exception as e:
        print(f"\n❌ Failed to save conversation: {e}")
        logger.error(f"Error saving conversation to {save_path}: {e}", exc_info=True)


def handle_load_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /load command to load a conversation."""
    if not args:
        print("\n⚠️ Usage: /load <filename>")
        return
    filename = args[0]

    saved_conversations_dir = agent.config.get('SAVED_CONVERSATIONS_DIRECTORY')
    if isinstance(saved_conversations_dir, str):
        project_root = Path(__file__).resolve().parent.parent
        load_path_base = project_root / saved_conversations_dir
    elif isinstance(saved_conversations_dir, Path):
        load_path_base = saved_conversations_dir
    else:
        project_root = Path(__file__).resolve().parent.parent
        load_path_base = project_root / 'SAVED_CONVERSATIONS'

    load_path = load_path_base / filename
    if not load_path.is_file():
        print(f"\n❌ File not found: {load_path}")
        return

    try:
        with open(load_path, 'r') as f:
            load_state = json.load(f)

        reconstructed_history = []
        if 'conversation_history' in load_state and isinstance(load_state['conversation_history'], list):
            for item in load_state['conversation_history']:
                if isinstance(item, dict) and 'role' in item and 'parts' in item and isinstance(item['parts'], list):
                    # Ensure parts are correctly formed for google.genai.types.Content
                    valid_parts = [types.Part(text=part_text) for part_text in item['parts'] if isinstance(part_text, str)]
                    if valid_parts: # Only create content if there are valid parts
                        content = types.Content(role=item['role'], parts=valid_parts)
                        reconstructed_history.append(content)
                    else:
                        logger.warning(f"Skipping history item with no valid text parts: {item}")
                else:
                    logger.warning(f"Skipping invalid item in loaded history: {item}")
        else:
            logger.warning(f"'conversation_history' key missing or not a list in {filename}")

        agent.conversation_history = reconstructed_history
        agent.current_token_count = load_state.get('current_token_count', 0)
        agent.prompt_time_counts = load_state.get('prompt_time_counts', [0])
        agent.messages_per_interval = load_state.get('messages_per_interval', [0])
        agent._messages_this_interval = load_state.get('_messages_this_interval', 0)
        agent.active_files = [] # Files are not restored from save, user needs to re-add
        
        # Restore thinking budget and pending states
        agent.thinking_budget = load_state.get('thinking_budget', agent.config.get('default_thinking_budget', 256))
        agent.thinking_config = types.ThinkingConfig(thinking_budget=agent.thinking_budget) # Re-apply thinking config
        agent.pending_pdf_context = load_state.get('pending_pdf_context')
        
        # Restore model name if available, otherwise use current agent's default
        loaded_model_name = load_state.get('model_name', agent.model_name)
        if agent.model_name != loaded_model_name:
            logger.info(f"Loaded conversation used model '{loaded_model_name}'. Current agent model is '{agent.model_name}'.")
            # For simplicity, we'll use the agent's current model for the new chat session.
            # If strict model adherence from save file is needed, agent.model_name would need to be updated
            # and potentially the client/chat re-initialized if the model change is significant.

        if agent.client:
            agent.chat = agent.client.chats.create(model=agent.model_name, history=agent.conversation_history)
            print(f"\n📂 Loaded conversation from: {filename} (using model {agent.model_name})")
            if agent.pending_pdf_context: print("   Loaded pending PDF context is active.")
        else:
            print("\n❌ Client not configured. Cannot fully restore chat session from load.")
            # History and pending states are loaded, but chat object isn't live.

    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to load conversation: Invalid JSON in {filename}. Error: {e}")
        logger.error(f"JSON decode error loading {load_path}: {e}", exc_info=True)
    except Exception as e:
        print(f"\n❌ Failed to load conversation: {e}")
        logger.error(f"Error loading conversation from {load_path}: {e}", exc_info=True)


def handle_thinking_budget_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /thinking_budget command."""
    if len(args) == 1:
        try:
            new_budget = int(args[0])
            if 0 <= new_budget <= 24000: 
                agent.thinking_budget = new_budget
                agent.thinking_config = types.ThinkingConfig(thinking_budget=agent.thinking_budget)
                print(f"\n🧠 Thinking budget updated to: {agent.thinking_budget} tokens.")
            else:
                print("\n⚠️ Thinking budget must be between 0 and 24000.")
        except ValueError:
            print("\n⚠️ Invalid number format for thinking budget.")
    else:
        print("\n⚠️ Usage: /thinking_budget <number_of_tokens>")


def handle_reset_command(agent: 'CodeAgent'):
    """Handles the /reset command to clear chat history."""
    print("\n🎯 Resetting context and starting a new chat session...")
    if agent.client:
        agent.chat = agent.client.chats.create(model=agent.model_name, history=[])
    else:
        agent.chat = None 
        print("\n⚠️ Client not configured, cannot create new chat session. History cleared locally.")

    agent.conversation_history = []
    agent.current_token_count = 0
    agent.active_files = [] 
    agent.prompt_time_counts = [0]
    agent.messages_per_interval = [0]
    agent._messages_this_interval = 0
    agent.pending_pdf_context = None
    agent.pending_script_output = None 
    print("\n✅ Chat session and history cleared.")


def handle_clear_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /clear command to remove tokens from history."""
    if not args:
        print("\n⚠️ Usage: /clear <number_of_tokens>")
        return
    try:
        tokens_to_clear_target = int(args[0])
        if tokens_to_clear_target <= 0:
            print("\n⚠️ Number of tokens must be positive.")
            return

        if not agent.conversation_history:
            print("Chat history is already empty.")
            return

        if not agent.client: 
            print("\n⚠️ Gemini client not available. Cannot accurately clear tokens by count.")
            return

        logger.info(f"Attempting to clear approx. {tokens_to_clear_target} tokens from history.")
        
        new_history = list(agent.conversation_history)
        tokens_counted_for_removal = 0
        messages_removed_count = 0
        history_modified_by_clear = False

        while tokens_counted_for_removal < tokens_to_clear_target and new_history:
            first_message = new_history[0]
            message_tokens = 0
            try:
                if isinstance(first_message, types.Content):
                    message_tokens = agent.client.models.count_tokens(model=agent.model_name, contents=[first_message]).total_tokens
                else:
                    logger.warning(f"Skipping non-Content item in history for token counting: {type(first_message)}")
                    message_tokens = 75 
            except Exception as e_count:
                logger.error(f"Could not count tokens for a message during /clear: {e_count}. Using estimate.", exc_info=True)
                message_tokens = 75 

            tokens_counted_for_removal += message_tokens
            new_history.pop(0)
            messages_removed_count += 1
            history_modified_by_clear = True
        
        logger.info(f"After initial pass, {messages_removed_count} messages ({tokens_counted_for_removal} tokens) selected for removal.")

        additional_messages_removed_for_role = 0
        while new_history and new_history[0].role != "user":
            logger.warning("History after initial clear pass starts with a model turn. Removing additional leading model messages.")
            new_history.pop(0)
            messages_removed_count += 1
            additional_messages_removed_for_role += 1
            history_modified_by_clear = True
        if additional_messages_removed_for_role > 0:
            logger.info(f"Removed {additional_messages_removed_for_role} additional model messages to ensure user turn start.")

        if not history_modified_by_clear:
             print(f"No messages were cleared. Requested {tokens_to_clear_target} tokens might be less than the first message(s), history is empty, or history is too short.")
             return

        agent.conversation_history = new_history

        if not agent.conversation_history:
            agent.current_token_count = 0
            agent.prompt_time_counts = [0]
            agent.messages_per_interval = [0]
        else:
            try:
                agent.current_token_count = agent.client.models.count_tokens(
                    model=agent.model_name,
                    contents=agent.conversation_history
                ).total_tokens
            except Exception as e_recount:
                logger.error(f"Error recounting tokens after /clear: {e_recount}. Token count may be inaccurate.", exc_info=True)
                agent.current_token_count = -1 # Indicate error
            agent.prompt_time_counts = [0, agent.current_token_count if agent.current_token_count != -1 else 0] 
            agent.messages_per_interval = [0, len(agent.conversation_history)]
        
        agent._messages_this_interval = 0 

        try:
            agent.chat = agent.client.chats.create(model=agent.model_name, history=agent.conversation_history)
            logger.info("Chat session re-initialized after /clear operation.")
            print(f"\n✅ Cleared {messages_removed_count} message(s) (approx. {tokens_counted_for_removal} tokens counted). "
                  f"New total tokens: {agent.current_token_count if agent.current_token_count != -1 else 'Error counting'}")
        except Exception as e_chat_reinit:
            logger.error(f"Error re-initializing chat after /clear: {e_chat_reinit}", exc_info=True)
            print(f"\n⚠️ Error re-initializing chat session: {e_chat_reinit}. History updated, but chat object may be stale.")
            
    except ValueError:
        print("\n⚠️ Invalid number format for tokens.")
    except Exception as e:
        print(f"\n⚠️ Error processing /clear command: {e}")
        logger.error("Error in /clear command", exc_info=True)


def handle_run_script_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /run_script command."""
    if len(args) >= 2: 
        script_type = args[0].lower()
        script_path = args[1]
        script_arguments = args[2:]
        if script_type in ["python", "shell"]:
            agent._handle_run_script_command(script_type, script_path, script_arguments)
        else:
            print("\n⚠️ Invalid script type. Must be 'python' or 'shell'. Usage: /run_script <python|shell> <script_path> [args...]")
    else:
        print("\n⚠️ Usage: /run_script <python|shell> <script_path> [args...]")


def handle_cancel_task_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /cancel command."""
    if args:
        agent._handle_cancel_command(args[0]) 
    else:
        print("\n⚠️ Usage: /cancel <task_id>")


def handle_list_tasks_command(agent: 'CodeAgent'):
    """Handles the /tasks command."""
    agent._handle_list_tasks_command()


def handle_history_command(agent: 'CodeAgent', args: List[str]):
    """Handles the /history command to display parts of the conversation history."""
    if not agent.client or not hasattr(agent.client, 'models'):
        print("\n⚠️ Gemini client not available for token counting. Cannot display history by token count.")
        return

    if not agent.conversation_history:
        print("\nℹ️ Conversation history is empty.")
        return

    mode = None
    num_tokens_target = 0

    if len(args) == 2:
        if args[0].lower() == '--head':
            mode = 'head'
        elif args[0].lower() == '--tail':
            mode = 'tail'
        
        try:
            num_tokens_target = int(args[1])
            if num_tokens_target <= 0:
                raise ValueError("Token count must be positive.")
        except ValueError:
            print("\n⚠️ Invalid number of tokens. Must be a positive integer.")
            print("   Usage: /history --head <n_tokens> OR /history --tail <n_tokens>")
            return
    
    if not mode or num_tokens_target == 0:
        print("\n⚠️ Invalid usage.")
        print("   Usage: /history --head <n_tokens> OR /history --tail <n_tokens>")
        return

    print(f"\n📜 Displaying {mode} {num_tokens_target} tokens from history (approximate):")
    
    history_to_scan = list(agent.conversation_history) # Make a copy
    if mode == 'tail':
        history_to_scan.reverse() # Iterate from the end for tail

    tokens_counted = 0
    messages_to_display = []

    for message_content in history_to_scan:
        if not isinstance(message_content, types.Content):
            logger.warning(f"Skipping non-Content item in history during /history: {type(message_content)}")
            continue

        try:
            message_tokens = agent.client.models.count_tokens(
                model=agent.model_name, 
                contents=[message_content]
            ).total_tokens
        except Exception as e:
            logger.error(f"Error counting tokens for history message: {e}")
            # Fallback: assign an average token count or skip
            message_tokens = 75 # Arbitrary average

        if tokens_counted + message_tokens > num_tokens_target and messages_to_display:
            # If adding this message exceeds target AND we already have some messages, stop.
            # This ensures we don't add a very large message if it's the first one that fits.
            break
        
        messages_to_display.append(message_content)
        tokens_counted += message_tokens

        if tokens_counted >= num_tokens_target:
            break
            
    if mode == 'tail':
        messages_to_display.reverse() # Reverse back to original order for printing

    if not messages_to_display:
        print("\n   No messages found within the specified token limit or history is effectively empty.")
        return

    for msg in messages_to_display:
        role_emoji = "👤" if msg.role == "user" else "🤖"
        # Assuming messages have a simple text part for this display
        text_parts = [part.text for part in msg.parts if hasattr(part, 'text') and part.text is not None]
        display_text = " ".join(text_parts)
        # Truncate long messages for display here if necessary
        # For now, printing full text of selected messages
        print(f"  {role_emoji} [{msg.role.upper()}]: {display_text[:500]}{ '...' if len(display_text) > 500 else ''}")
    
    print(f"\n(Displayed approx. {tokens_counted} tokens from {len(messages_to_display)} messages)")


def handle_toggle_verbose_command(agent: 'CodeAgent'):
    """Toggles verbose logging on/off."""
    import logging
    root_logger = logging.getLogger()
    
    if root_logger.level == logging.WARNING:
        # Currently not verbose, switch to verbose (INFO)
        root_logger.setLevel(logging.INFO)
        print("\n🔊 Verbose logging enabled (INFO level)")
    else:
        # Currently verbose, switch to non-verbose (WARNING)
        root_logger.setLevel(logging.WARNING)
        print("\n🔇 Verbose logging disabled (WARNING level)")


# --- Command Dispatcher ---
COMMAND_HANDLERS: Dict[str, callable] = {
    "/pdf": handle_pdf_command,
    "/prompt": handle_prompt_command,
    "/save": handle_save_command,
    "/load": handle_load_command,
    "/thinking_budget": handle_thinking_budget_command,
    "/reset": handle_reset_command,
    "/clear": handle_clear_command,
    "/run_script": handle_run_script_command,
    "/cancel": handle_cancel_task_command,
    "/tasks": handle_list_tasks_command,
    "/help": handle_help_command, 
    "/history": handle_history_command,
    "/toggle_verbose": handle_toggle_verbose_command,
} 