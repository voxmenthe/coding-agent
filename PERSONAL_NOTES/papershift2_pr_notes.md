main.py

# Line 365 changed block:
- 'conversation_history': [c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.conversation_history],
+ 'conversation_history': [],

# Line 394 (new section):
                    # Convert Content objects to serializable dictionaries
                    serializable_history = []
                    for content in self.conversation_history:
                        # Ensure parts is a list of strings
                        parts_text = [part.text for part in content.parts if hasattr(part, 'text') and part.text is not None]
                        serializable_history.append({
                            'role': content.role,
                            'parts': parts_text
                        })
                    save_state['conversation_history'] = serializable_history

# Line 429 changed block:
- self.conversation_history = load_state.get('conversation_history', [])
#replaced by:
                       # Restore state
                        reconstructed_history = []
                        if 'conversation_history' in load_state and isinstance(load_state['conversation_history'], list):
                            for item in load_state['conversation_history']:
                                if isinstance(item, dict) and 'role' in item and 'parts' in item and isinstance(item['parts'], list):
                                    # Recreate Parts from the list of text strings
                                    parts = [glm.Part(text=part_text) for part_text in item['parts'] if isinstance(part_text, str)]
                                    # Recreate Content object
                                    content = glm.Content(role=item['role'], parts=parts)
                                    reconstructed_history.append(content)
                                else:
                                    logger.warning(f"Skipping invalid item in loaded history: {item}")
                        else:
                             logger.warning(f"'conversation_history' key missing or not a list in {filename}")

                        self.conversation_history = reconstructed_history

# Line 611 changed block:
                - new_user_content = types.Content(parts=[types.Part(text=user_input)], role="user")
                + new_user_content = glm.Content(parts=[glm.Part(text=message_to_send)], role="user")


# Line 626 changed blcok:
-                # --- Update manual history and calculate new token count AFTER response --- 
-                agent_response_content = None
-                response_text = "" # Initialize empty response text
-                if response.candidates and response.candidates[0].content:
-                    agent_response_content = response.candidates[0].content
-                    # Ensure we extract text even if other parts exist (e.g., tool calls)
-                    if agent_response_content.parts:
-                         # Simple concatenation of text parts for history
-                         response_text = " ".join(p.text for p in agent_response_content.parts -if hasattr(p, 'text'))
-                    self.conversation_history.append(agent_response_content)

+               agent_response_text = ""
+                if response.candidates and response.candidates[0].content:
+                     agent_response_content = response.candidates[0].content
+                     # Extract text from all parts for printing
+                     agent_response_text = " ".join(p.text for p in agent_response_content.parts if hasattr(p, 'text'))
+ 
+                if agent_response_text: # Only append if there's text
+                     history_agent_content = glm.Content(role="model", parts=[glm.Part(text=agent_response_text)])
+                     self.conversation_history.append(history_agent_content)
+                     logger.debug(f"Appended model text response to history: Role={history_agent_content.role}")

# Line 643 (new section):
                # --- Detailed History Logging Before Token Count --- 
                logger.debug(f"Inspecting conversation_history (length: {len(self.conversation_history)}) before count_tokens:")
                history_seems_ok = True
                for i, content in enumerate(self.conversation_history):
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

# Line 786 - deleted pdf processing method pypdf
                # TODO: Implement fallback to pypdf if configured or if gemini fails?
                try:
                    if self.pdf_processing_method == 'pypdf':
                        from pypdf import PdfReader
                        reader = PdfReader(pdf_path)
                        extracted_text = ""
                        for page in reader.pages:
                            extracted_text += page.extract_text() + "\n"
                    if not extracted_text:
                        print("  ⚠️ Warning: PyPDF fallback extracted no text.")
                        extracted_text = f"Placeholder: No text extracted via PyPDF for {filename}" 
                except Exception as pypdf_err:
                    print(f"  ⚠️ PyPDF fallback failed: {pypdf_err}. Using basic placeholder.")
                    extracted_text = f"Placeholder: Error during PyPDF extraction for {filename}"

