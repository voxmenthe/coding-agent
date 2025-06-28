import logging
import re
from pathlib import Path
from typing import Iterable, Dict, Optional, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, NestedCompleter, Document, WordCompleter, CompleteEvent
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.keys import Keys
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from google.genai import types

from .autocomplete import PdfCompleter
from . import slashcommands

logger = logging.getLogger(__name__)

class LoggingCompleterWrapper(Completer):
    def __init__(self, wrapped_completer: Completer, name: str = "Wrapped"):
        self.wrapped_completer = wrapped_completer
        self.name = name

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        logger.info(f"[{self.name}] get_completions CALLED. Text: '{document.text_before_cursor}', UserInvoked: {complete_event.completion_requested}")
        raw_completions = list(self.wrapped_completer.get_completions(document, complete_event))
        logger.info(f"[{self.name}] Raw completions from wrapped: {[c.text for c in raw_completions]} for input '{document.text_before_cursor}'")
        yield from raw_completions

class CustomNestedCompleter(NestedCompleter):
    def __init__(self, options: Dict[str, Optional[Completer]], ignore_case: bool = True):
        super().__init__(options, ignore_case=ignore_case)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        line_text_before_cursor = document.current_line_before_cursor
        line_cursor_col = document.cursor_position_col
        words_info = list(re.finditer(r'\S+', line_text_before_cursor))
        command_segment_doc = None
        for i in range(len(words_info) - 1, -1, -1):
            word_match = words_info[i]
            word_text = word_match.group(0)
            word_start_col_in_line = word_match.start()
            word_end_col_in_line = word_match.end()
            is_cursor_engaged_with_word_in_line = (
                (line_cursor_col >= word_start_col_in_line and line_cursor_col <= word_end_col_in_line) or
                (line_cursor_col == word_end_col_in_line + 1 and line_text_before_cursor.endswith(' '))
            )
            if word_text.startswith('/') and is_cursor_engaged_with_word_in_line:
                segment_text_on_line = line_text_before_cursor[word_start_col_in_line:]
                segment_cursor_pos_in_line = line_cursor_col - word_start_col_in_line
                command_segment_doc = Document(text=segment_text_on_line, cursor_position=segment_cursor_pos_in_line)
                break
        if not command_segment_doc:
            yield from []
            return
        sub_text_before_cursor = command_segment_doc.text_before_cursor.lstrip()
        if ' ' not in sub_text_before_cursor:
            actual_keys = [k for k in self.options.keys() if isinstance(k, str)]
            first_word_completer = WordCompleter(words=actual_keys, ignore_case=self.ignore_case, match_middle=True, sentence=False)
            for c in first_word_completer.get_completions(command_segment_doc, complete_event):
                insert_text = c.text
                display_text = c.text
                if command_segment_doc.text.lstrip().startswith('/') and c.text.startswith('/') and len(c.text) > 1:
                    insert_text = c.text[1:]
                yield Completion(
                    text=insert_text,
                    start_position=c.start_position,
                    display=display_text,
                    display_meta=c.display_meta,
                    style=c.style,
                    selected_style=c.selected_style
                )
        else:
            for c in super().get_completions(command_segment_doc, complete_event):
                yield Completion(
                    text=c.text,
                    start_position=c.start_position,
                    display=c.display,
                    display_meta=c.display_meta,
                    style=c.style,
                    selected_style=c.selected_style
                )

def is_typing_slash_command_prefix(current_buffer):
    text_up_to_cursor = current_buffer.document.text_before_cursor
    if text_up_to_cursor.endswith('/'):
        return True
    match = re.search(r'(\S+)$', text_up_to_cursor)
    if match:
        current_segment = match.group(1)
        if current_segment.startswith('/'):
            return True
    return False

def _get_dynamic_prompt_message(agent):
    active_files_info = f" [{len(agent.active_files)} files]" if agent.active_files else ""
    token_info = f"({agent.current_token_count})"
    return HTML(
        f'<ansiblue>🔵 You</ansiblue> '
        f'<ansicyan>{token_info}</ansicyan>'
        f'<ansigreen>{active_files_info}</ansigreen>: '
    )

def _get_continuation_prompt(width, line_number, is_soft_wrap):
    if is_soft_wrap:
        return ' ' * 2
    return HTML('<ansiyellow>│ </ansiyellow>')

def _get_bottom_toolbar():
    return HTML(
        '<ansigray>'
        '[Alt+Enter] Submit │ [Enter] New line │ [Ctrl+D] Quit │ '
        '[↑↓] History │ [Tab] Complete'
        '</ansigray>'
    )

def _create_enhanced_prompt_session(agent, command_completer, history):
    kb = KeyBindings()

    @kb.add('enter')
    def _(event):
        event.current_buffer.insert_text('\n')

    @kb.add('escape', 'enter')
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add('c-j')
    def _(event):
        event.current_buffer.validate_and_handle()

    @kb.add('c-d')
    def _(event):
        if not event.current_buffer.text.strip():
            event.app.exit(result='exit_eof')

    @kb.add(Keys.Any, filter=Condition(lambda: is_typing_slash_command_prefix(get_app().current_buffer)))
    def _handle_slash_command_typing(event):
        event.current_buffer.insert_text(event.data)
        event.app.current_buffer.start_completion(select_first=False)

    return PromptSession(
        message=lambda: _get_dynamic_prompt_message(agent),
        multiline=True,
        wrap_lines=True,
        mouse_support=True,
        complete_style=CompleteStyle.MULTI_COLUMN,
        completer=command_completer,
        history=history,
        key_bindings=kb,
        auto_suggest=AutoSuggestFromHistory(),
        search_ignore_case=True,
        prompt_continuation=_get_continuation_prompt,
        bottom_toolbar=_get_bottom_toolbar,
        complete_while_typing=True,
        enable_history_search=False,
    )

def run_cli(agent):
    """Interactive CLI loop for CodeAgent."""
    if not agent.client:
        print("\n\u274c Client not configured. Exiting.")
        return

    agent.print_initial_help() # Use the new brief help message

    # Set initial thinking budget from default/config
    agent.thinking_config = types.ThinkingConfig(thinking_budget=agent.thinking_budget)
    print(f"\n🧠 Initial thinking budget set to: {agent.thinking_budget} tokens.")

    # Print multi-line editing instructions
    print("\n📝 Multi-line editing enabled:")
    print("   • Press [Enter] to create new lines")
    print("   • Press [Alt+Enter] or [Ctrl+J] to submit")
    print("   • Use mouse to select text and position cursor")
    print("   • Press [↑↓] to navigate history")

    # Define slash commands and setup nested completer
    # The keys in slash_command_completer_map should match COMMAND_HANDLERS keys
    slash_command_completer_map: Dict[str, Optional[Completer]] = {
        '/reset': None,
        '/exit': None,
        '/q': None,
        '/clear': None, # Takes an arg, but simple WordCompleter not ideal
        '/save': None,  # Takes an optional arg
        '/thinking_budget': None, # Takes an arg
        '/cancel': None, # Takes an arg (task_id)
        '/tasks': None,
        '/run_script': None, # Complex args, PathCompleter for script_path might be good
        '/help': None,
        '/history': None, # New history command
         # /pdf and /load have specific completers
    }

    # PDF file completer
    # PdfCompleter now takes 'self' (the CodeAgent instance)
    slash_command_completer_map['/pdf'] = PdfCompleter(agent)

    # Saved conversations completer
    saved_conversations_dir_path = agent.config.get('SAVED_CONVERSATIONS_DIRECTORY')
    saved_files = []
    if saved_conversations_dir_path and isinstance(saved_conversations_dir_path, Path) and saved_conversations_dir_path.is_dir():
         saved_files = [f.name for f in saved_conversations_dir_path.glob('*.json') if f.is_file()]
    elif isinstance(saved_conversations_dir_path, str): 
         logger.warning("SAVED_CONVERSATIONS_DIRECTORY was a string, expected Path. Attempting to resolve.")
         try:
              # Assuming project_root is defined if this fallback is hit, or use Path(__file__).parent.parent
              project_root = Path(__file__).parent.parent 
              resolved_path = project_root / saved_conversations_dir_path
              if resolved_path.is_dir():
                   saved_files = [f.name for f in resolved_path.glob('*.json') if f.is_file()]
         except Exception as e:
              logger.error(f"Error resolving string path for saved conversations: {e}")
    else: 
         default_save_dir = Path(__file__).parent.parent / 'SAVED_CONVERSATIONS/'
         if default_save_dir.is_dir():
              saved_files = [f.name for f in default_save_dir.glob('*.json') if f.is_file()]
    slash_command_completer_map['/load'] = WordCompleter(saved_files, ignore_case=True)

    # Prompt name completer
    slash_command_completer_map['/prompt'] = WordCompleter(agent._list_available_prompts(), ignore_case=True)
    
    # Add all command names from COMMAND_HANDLERS to the completer if not already specified
    for cmd_name in slashcommands.COMMAND_HANDLERS.keys():
        if cmd_name not in slash_command_completer_map:
            slash_command_completer_map[cmd_name] = None

    # Create the CustomNestedCompleter instance
    command_completer_instance = CustomNestedCompleter(
        options=slash_command_completer_map,
        ignore_case=True
    )

    logger.debug(f"Using CustomNestedCompleter with options: {list(slash_command_completer_map.keys())}")

    history = InMemoryHistory()
    # Create enhanced session
    session = agent._create_enhanced_prompt_session(command_completer_instance, history)

    while True:
        try:
            # ─── 1 · house‑keeping before we prompt ──────────────────────────
            agent.prompt_time_counts.append(agent.current_token_count)
            agent.messages_per_interval.append(agent._messages_this_interval)
            agent._messages_this_interval = 0

            # Get multi-line input with the enhanced session
            # The prompt message is now handled by _get_dynamic_prompt_message
            # Prefill logic needs to be integrated with PromptSession's `default` if needed, or handled before prompt.
            # For now, we simplify and remove direct prefill_text here as the plan's session.prompt() is simpler.
            # If prefill is critical, it should be passed to session.prompt(default=prefill_text)
            
            # Check if there's content to prefill from /prompt command
            current_prefill_text = ""
            if agent.prefill_prompt_content:
                current_prefill_text = agent.prefill_prompt_content
                agent.prefill_prompt_content = None # Clear after retrieving
            
            try:
                user_input = session.prompt(default=current_prefill_text).strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:  # Ctrl+D (or result from custom keybinding)
                print("\n👋 Goodbye!") # Or handle 'exit_eof' if needed
                break

            # ─── 2 · trivial exits / empty line ─────────────────────────────
            # Handle exit commands (already covered by Ctrl+D and KeyboardInterrupt for session)
            # The custom Ctrl+D binding in _create_enhanced_prompt_session handles empty line exit.
            # If user types 'exit' or 'quit' it will be processed as a command or message.
            # The plan's version of this check is slightly different, let's ensure it's robust.
            if user_input.lower() in {"exit", "quit"}: # This is from the plan, simpler than existing /exit /q
                print("\n👋 Goodbye!")
                break
            
            # If Ctrl+D was used on an empty line, app.exit(result='exit_eof') is called.
            # The prompt loop will break due to EOFError in that case.
            # No specific check for 'exit_eof' needed here if EOFError is caught.

            if not user_input:
                agent.prompt_time_counts.pop()
                agent.messages_per_interval.pop()
                continue

            # --- 3 · Command Parsing and Dispatch ---
            command_parts = user_input.split()
            command_name = command_parts[0].lower()
            command_args = command_parts[1:]

            handler: Optional[Callable] = slashcommands.COMMAND_HANDLERS.get(command_name)

            if handler:
                # Determine if the handler takes arguments.
                # This is a simple check; more robust would be `inspect.signature`.
                # For now, we assume handlers like /reset, /tasks, /help don't need args.
                # Updated logic to pass session to /prompt handler
                if command_name in ["/reset", "/tasks", "/help"]:
                    handler(agent) # Call with agent only
                elif command_name == "/prompt":
                    handler(agent, session, command_args) # Call /prompt with agent, session, args
                elif command_name == "/save" and not command_args:
                    handler(agent, []) # Call /save with agent and empty args list
                else:
                    # Default for commands taking agent and args
                    handler(agent, command_args)

                continue # Command handled, loop to next prompt
            
            # --- If not a known slash command, proceed as LLM message ---
            agent._messages_this_interval += 1 
            message_to_send = user_input 

            pdf_context_was_included = False
            script_output_was_included = False # New flag
             
            # Check for PDF context AFTER prompt (so prompt comes first)
            if agent.pending_pdf_context:
                print("[Including context from previously processed PDF in this message.]\n")
                message_to_send = f"{agent.pending_pdf_context}\n\n{message_to_send}"
                pdf_context_was_included = True
            
            # Prepend script output AFTER PDF but BEFORE loaded prompt
            if agent.pending_script_output:
                print("[Including output from previously run script in this message.]\n")
                # Task name might not be easily available here, use a generic header
                message_to_send = f"OUTPUT FROM EXECUTED SCRIPT:\n---\n{agent.pending_script_output}\n---\n\n{message_to_send}"
                script_output_was_included = True
            
            # --- Log message details before sending --- 
            pdf_context_len_before_send = len(agent.pending_pdf_context) if agent.pending_pdf_context and pdf_context_was_included else 0
            script_output_len_before_send = len(agent.pending_script_output) if agent.pending_script_output and script_output_was_included else 0
            final_message_len = len(message_to_send)
            logger.info(f"Preparing to send message.")
            logger.info(f"  - Original user input length: {len(user_input)}")
            logger.info(f"  - Included pending PDF context length: {pdf_context_len_before_send}")
            logger.info(f"  - Included pending script output length: {script_output_len_before_send}")
            logger.info(f"  - Final message_to_send length: {final_message_len}")
            # Log snippets for verification
            if final_message_len > 200:
                logger.info(f"  - Final message start: {message_to_send[:100]}...")
                logger.info(f"  - Final message end: ...{message_to_send[-100:]}")

            # --- Prepare message content (Text + Files) ---
            message_content = [message_to_send]
            if agent.active_files:
                message_content.extend(agent.active_files)
                if agent.config.get('verbose', False):
                    print(f"\n📎 Attaching {len(agent.active_files)} files to the prompt:")
                    for f in agent.active_files:
                        print(f"   - {f.display_name} ({f.name})")

            # --- Update manual history (for token counting ONLY - Use Text Only) --- 
            new_user_content =types.Content(parts=[types.Part(text=message_to_send)], role="user")
            agent.conversation_history.append(new_user_content)

            # --- Send Message --- 
            print("\n⏳ Sending message and processing...")
            # Prepare tool configuration **inside the loop** to use the latest budget
            tool_config = types.GenerateContentConfig(
                tools=agent.tool_functions, 
                thinking_config=agent.thinking_config
            )

            # Send message using the chat object's send_message method
            # Pass the potentially combined list of text and files
            response = agent.chat.send_message(
                message=message_content, # Pass the list here
                config=tool_config
            )

            agent_response_text = ""
            if response.candidates and response.candidates[0].content:
                agent_parts = response.candidates[0].content.parts
                agent_response_text = " ".join(p.text for p in agent_parts
                                               if hasattr(p, "text"))

            if agent_response_text:
                hist_agent_content = types.Content(role="model",
                                                 parts=[types.Part(text=agent_response_text)])
                agent.conversation_history.append(hist_agent_content)

            print(f"\n🟢 \x1b[92mAgent:\x1b[0m {agent_response_text or '[No response text]'}")

            # --- Detailed History Logging Before Token Count --- 
            logger.debug(f"Inspecting conversation_history (length: {len(agent.conversation_history)}) before count_tokens:")
            history_seems_ok = True
            for i, content in enumerate(agent.conversation_history):
                logger.debug(f"  [{i}] Role: {getattr(content, 'role', 'N/A')}")
                if hasattr(content, 'parts'):
                    for j, part in enumerate(content.parts):
                        part_type = type(part)
                        part_info = f"Part {j}: Type={part_type.__name__}"
                        if hasattr(part, 'text'):
                            part_info += f", Text='{part.text[:50]}...'"
                        elif hasattr(part, 'file_data'):
                            part_info += f", FileData URI='{getattr(part.file_data, 'file_uri', 'N/A')}'"
                            history_seems_ok = False # Found a file part!
                            logger.error(f"    🚨 ERROR: Found unexpected file_data part in history for token counting: {part_info}")
                        elif hasattr(part, 'function_call'):
                            part_info += f", FunctionCall Name='{getattr(part.function_call, 'name', 'N/A')}'"
                            history_seems_ok = False # Found a function call part!
                            logger.error(f"    🚨 ERROR: Found unexpected function_call part in history for token counting: {part_info}")
                        else:
                            # Log other unexpected part types
                            history_seems_ok = False
                            logger.error(f"    🚨 ERROR: Found unexpected part type in history for token counting: {part_info}")
                        logger.debug(f"    {part_info}")
                else:
                    logger.warning(f"  [{i}] Content object has no 'parts' attribute.")
            if history_seems_ok:
                logger.debug("History inspection passed: Only text parts found.")
            else:
                logger.error("History inspection FAILED: Non-text parts found. Token counting will likely fail.")
            # --- End Detailed History Logging --- 

            # Calculate and display token count using client.models
            try:
                token_info = agent.client.models.count_tokens(
                    model=agent.model_name,
                    contents=agent.conversation_history
                )
                agent.current_token_count = token_info.total_tokens
                print(f"\n[Token Count: {agent.current_token_count}]")
            except Exception as count_err:
                logger.error(f"Error calculating token count: {count_err}", exc_info=True)
                print("🚨 Error: Failed to calculate token count.")

            # --- NOW clear contexts that were actually sent --- 
            if pdf_context_was_included:
                agent.pending_pdf_context = None
                logger.info("Cleared pending_pdf_context after sending to LLM.")
            if script_output_was_included:
                agent.pending_script_output = None
                logger.info("Cleared pending_script_output after sending to LLM.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n🔴 An error occurred during interaction: {e}")
            traceback.print_exc()
            # Potentially add a small delay or a prompt to continue/exit here
