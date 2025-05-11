import os
import re
from pathlib import Path
from datetime import datetime
from prompt_toolkit.completion import Completer, Completion
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict, Any

if TYPE_CHECKING:
    from .main import CodeAgent  # Import CodeAgent for type hinting

class PdfCompleter(Completer):
    """
    A custom completer for the /pdf command that supports filtering and sorting.
    """
    def __init__(self, agent: 'CodeAgent'):
        self.agent = agent
        # These attributes for storing state between completions are not typically how
        # prompt_toolkit completers work, as get_completions should be stateless
        # or derive state from the document. For now, they are unused in get_completions.
        # self.current_sort_attribute: str = 'name'
        # self.current_sort_ascending: bool = True
        # self.current_filter_substring: Optional[str] = None

    def _get_all_pdf_details(self) -> List[Tuple[str, Path, float]]:
        """
        Fetches all PDF files from the configured directory.
        Returns a list of tuples: (filename, path_object, last_modified_timestamp).
        """
        pdf_details = []
        if self.agent.pdfs_dir_abs_path and self.agent.pdfs_dir_abs_path.is_dir():
            try:
                for f_path in self.agent.pdfs_dir_abs_path.glob('*.pdf'):
                    if f_path.is_file():
                        try:
                            timestamp = f_path.stat().st_mtime
                            pdf_details.append((f_path.name, f_path, timestamp))
                        except OSError:
                            # Could log error if specific file stat fails
                            pass # pragma: no cover
            except Exception as e:
                # Could log error if globbing fails
                pass # pragma: no cover
        return pdf_details

    def _filter_pdfs_by_name_substring(
        self, pdf_details: List[Tuple[str, Path, float]], substring: Optional[str]
    ) -> List[Tuple[str, Path, float]]:
        """Filters PDF details by a case-insensitive filename substring."""
        if not substring:
            return pdf_details
        return [
            detail for detail in pdf_details
            if substring.lower() in detail[0].lower()
        ]

    def _sort_pdfs_by_attribute(
        self, pdf_details: List[Tuple[str, Path, float]], attribute: str, ascending: bool
    ) -> List[Tuple[str, Path, float]]:
        """Sorts PDF details by attribute (name or time)."""
        if attribute == 'time':
            return sorted(pdf_details, key=lambda x: x[2], reverse=not ascending)
        # Default to sorting by name
        return sorted(pdf_details, key=lambda x: x[0].lower(), reverse=not ascending)

    def get_completions(self, document, complete_event):
        # document.text_before_cursor contains the text *after* "/pdf" (and potentially a space)
        # as provided by NestedCompleter.
        # Example: if user typed "/pdf --filter foo b", text_before_cursor is " --filter foo b"
        # If user typed "/pdf ", text_before_cursor can be " "
        # If user typed "/pdf", text_before_cursor can be ""
        command_args_text = document.text_before_cursor

        # For parsing arguments and determining `words`.
        # Example: " --filter foo b" -> "--filter foo b"
        # Example: " " -> ""
        # Example: "" -> ""
        parsed_command_args = command_args_text.lstrip()
        words = parsed_command_args.split() # Args like ['--filter', 'foo', 'b'] or []

        # current_word is the word at the cursor in the original command_args_text.
        # Example: "/pdf --filter foo b<TAB>" -> current_word = "b"
        # Example: "/pdf --filter foo <TAB>" -> current_word = "" (if cursor after space)
        # Example: "/pdf --s<TAB>" -> current_word = "--s"
        current_word = document.get_word_before_cursor()

        # --- Parse current arguments to set filter and sort state ---
        temp_sort_attribute = 'name'
        temp_sort_ascending = True
        temp_filter_substring = None
        filename_parts_typed = [] # Parts that are likely part of a filename

        i = 0
        while i < len(words):
            word_val = words[i]
            is_current_word_being_parsed = (word_val == current_word and not command_args_text.endswith(' '))
            
            if word_val == '--sort':
                if i + 1 < len(words):
                    # Check if next word is also the current word we are trying to complete for sort type
                    if is_current_word_being_parsed and i == len(words) -1 : break # stop parsing, current_word is --sort
                    if words[i+1] == current_word and not command_args_text.endswith(' ') and i+1 == len(words)-1: break

                    sort_type_arg = words[i+1].lower()
                    if sort_type_arg in ['alpha', 'name']:
                        temp_sort_attribute = 'name'
                    elif sort_type_arg == 'time':
                        temp_sort_attribute = 'time'
                    else: # Invalid sort type, could be part of a filename if not the current word
                        if not (words[i+1] == current_word and i+1 == len(words)-1 and not command_args_text.endswith(' ')):
                             filename_parts_typed.append(word_val) # add --sort as filename part
                             # filename_parts_typed.append(words[i+1]) # add invalid sort_type_arg as filename part
                        i += 1 # effectively consume word_val by moving to next
                        continue
                    i += 1 # consume sort type arg

                    if i + 1 < len(words): # Check for A/D
                        if is_current_word_being_parsed and i == len(words) -1: break # current_word is sort_type_arg
                        if words[i+1] == current_word and not command_args_text.endswith(' ') and i+1 == len(words)-1: break

                        sort_order_arg = words[i+1].upper()
                        if sort_order_arg == 'A':
                            temp_sort_ascending = True
                            i += 1
                        elif sort_order_arg == 'D':
                            temp_sort_ascending = False
                            i += 1
                        # If not A or D, it might be the start of a filename or next option
                i += 1 # consume --sort or its argument or move to next word if --sort is last
                continue
            elif word_val == '--filter':
                if i + 1 < len(words):
                    if is_current_word_being_parsed and i == len(words) -1: break # current_word is --filter
                    # if the filter value is the current word, don't parse it as a complete filter value yet
                    if words[i+1] == current_word and not command_args_text.endswith(' ') and i+1 == len(words)-1: break 
                    temp_filter_substring = words[i+1]
                    i += 1 # consume filter substring
                i += 1 # consume --filter or its argument or move to next word if --filter is last
                continue
            else: # Not an option, assume it's part of a filename
                # Add to filename_parts_typed only if it's a fully typed word (not the one being completed)
                if not (word_val == current_word and i == len(words) -1 and not command_args_text.endswith(' ')):
                    filename_parts_typed.append(word_val)
            i += 1
        
        # --- Determine completion context (what part of the command is being typed) ---
        is_completing_option_name = current_word.startswith('--')
        is_completing_sort_type = False
        is_completing_sort_order = False
        is_completing_filter_value = False # User types this freely, but we need to know for context

        if not is_completing_option_name and words:
            idx_of_current_word_in_words = -1
            if current_word in words:
                try:
                    idx_of_current_word_in_words = len(words) - 1 - words[::-1].index(current_word)
                except ValueError:
                    pass # current_word is not in words (e.g. new partial word)

            if idx_of_current_word_in_words > 0:
                prev_word_in_struct = words[idx_of_current_word_in_words - 1]
                if prev_word_in_struct == '--sort':
                    is_completing_sort_type = True
                elif prev_word_in_struct == '--filter':
                    is_completing_filter_value = True
                elif prev_word_in_struct in ['alpha', 'name', 'time'] and \
idx_of_current_word_in_words > 1 and words[idx_of_current_word_in_words - 2] == '--sort':
                    is_completing_sort_order = True
            elif not current_word and words: # current_word is empty (e.g. after a space)
                last_full_word = words[-1]
                if last_full_word == '--sort':
                    is_completing_sort_type = True
                elif last_full_word == '--filter':
                    is_completing_filter_value = True
                elif last_full_word in ['alpha', 'name', 'time'] and len(words) >= 2 and words[-2] == '--sort':
                    is_completing_sort_order = True
        
        # --- Determine current_filename_prefix ---
        current_filename_prefix = "" # Default to empty
        # Only set a filename prefix if we are NOT completing an option name, sort type, or sort order.
        if not (is_completing_option_name or is_completing_sort_type or is_completing_sort_order):
            if not is_completing_filter_value:
                 current_filename_prefix = current_word

        # --- Fetch and process PDFs ---
        all_pdf_details = self._get_all_pdf_details()
        # temp_filter_substring is determined by fully parsed arguments.
        # If we are currently *typing* the filter substring, we should use current_word for live filtering.
        active_filter_text = temp_filter_substring
        if is_completing_filter_value and current_word:
            active_filter_text = current_word
        
        filtered_pdfs = self._filter_pdfs_by_name_substring(all_pdf_details, active_filter_text)
        sorted_pdfs = self._sort_pdfs_by_attribute(filtered_pdfs, temp_sort_attribute, temp_sort_ascending)
        
        # --- Generate Completions ---
        if is_completing_option_name:
            if '--sort'.startswith(current_word):
                yield Completion('--sort ', start_position=-len(current_word), display_meta='Sort PDF list')
            if '--filter'.startswith(current_word):
                yield Completion('--filter ', start_position=-len(current_word), display_meta='Filter PDF list by name')
            return

        if is_completing_sort_type:
            if 'alpha'.startswith(current_word):
                yield Completion('alpha ', start_position=-len(current_word), display_meta='Sort by name')
            if 'time'.startswith(current_word):
                yield Completion('time ', start_position=-len(current_word), display_meta='Sort by modification time')
            # After suggesting sort types, also list files with this potential sort applied
            # effective_sort_attr = 'name' if 'alpha'.startswith(current_word) else 'time' if 'time'.startswith(current_word) else temp_sort_attribute
            # for fname, _, _ in self._sort_pdfs_by_attribute(filtered_pdfs, effective_sort_attr, temp_sort_ascending):
            #     if fname.lower().startswith(""): # Show all that match current filter
            #         yield Completion(fname, start_position=len(current_word) if current_word else 0, display_meta=f"File (sorted by {effective_sort_attr})")
            return

        if is_completing_sort_order:
            if 'A'.startswith(current_word.upper()):
                yield Completion('A ', start_position=-len(current_word), display_meta='Ascending')
            if 'D'.startswith(current_word.upper()):
                yield Completion('D ', start_position=-len(current_word), display_meta='Descending')
            # Also list files with this potential sort order applied
            # sort_type_word = words[idx_of_current_word_in_words -1] if idx_of_current_word_in_words != -1 else words[-1]
            # effective_sort_attr = 'name' if sort_type_word in ['alpha', 'name'] else 'time'
            # effective_sort_asc = True if 'A'.startswith(current_word.upper()) else False if 'D'.startswith(current_word.upper()) else temp_sort_ascending
            # for fname, _, _ in self._sort_pdfs_by_attribute(filtered_pdfs, effective_sort_attr, effective_sort_asc):
            #      if fname.lower().startswith(""):
            #         yield Completion(fname, start_position=len(current_word) if current_word else 0, display_meta=f"File (sorted {effective_sort_attr} {'ASC' if effective_sort_asc else 'DESC'})")
            return

        # Default: Complete filenames and, if appropriate, base options like --sort, --filter.
        # This is the fallback if no specific option part is being actively typed.
        # current_filename_prefix is what the user has typed for the filename so far.

        # Offer base options if current_word is empty (i.e., after a space where a new arg can start)
        # or if current_word itself could be the start of an option, AND the option hasn't been used.
        
        # Determine if --sort or --filter have been fully processed already
        # We check this by seeing if temp_sort_attribute was changed from default OR if a sort order was specified
        # For filter, we check if temp_filter_substring was set.
        sort_option_already_used = any(word == '--sort' for word in words if word != current_word) # A bit more direct
        filter_option_already_used = any(word == '--filter' for word in words if word != current_word)

        if not current_word or current_word.startswith('-'): # After a space, or starting to type an option
            if not sort_option_already_used and '--sort'.startswith(current_word):
                 yield Completion('--sort ', start_position=-len(current_word), display_meta='Sort PDF list')
            if not filter_option_already_used and '--filter'.startswith(current_word):
                 yield Completion('--filter ', start_position=-len(current_word), display_meta='Filter PDF list by name')
        
        # Always offer matching filenames based on the current prefix and parsed arguments
        for fname, fpath, ftime in sorted_pdfs:
            if fname.lower().startswith(current_filename_prefix.lower()):
                display_meta_text = 'PDF File'
                if temp_sort_attribute == 'time':
                    try:
                        formatted_date = datetime.fromtimestamp(ftime).strftime('%Y%m%d')
                        display_meta_text = f'{formatted_date} - PDF File'
                    except ValueError: # Handle potential errors with timestamp
                        pass # Stick to default meta text

                yield Completion(
                    fname,
                    start_position=-len(current_filename_prefix),
                    display_meta=display_meta_text
                )
        return 