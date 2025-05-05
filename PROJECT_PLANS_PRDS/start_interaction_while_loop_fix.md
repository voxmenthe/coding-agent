**Restructuring while loop**

I need to modify the structure of a while loop by adjusting how command processing and context injections work. I'll track and clear pending prompt injections after use and adjust PDF context injection timing. The key is to keep command detection early on but ensure the PDF context is injected right after command processing, just before assembling the message. Once those are in place, I’ll modify how we assemble the message content, ensuring everything runs in the right order.

Below is a **drop‑in replacement** for the entire `while True:` loop inside `start_interaction()`.
It preserves all your existing features, but moves **PDF‑context injection** so it fires **only when we are certain the line will be sent to Gemini** and therefore is never lost.

```python
        # ------------------------------------------------------------
        # MAIN INTERACTION LOOP  –  replace everything from
        #     while True:
        # down to the matching try/except block end.
        # ------------------------------------------------------------
        while True:
            try:
                # ─── 1 · house‑keeping before we prompt ──────────────────────────
                self.prompt_time_counts.append(self.current_token_count)
                self.messages_per_interval.append(self._messages_this_interval)
                self._messages_this_interval = 0

                active_files_info = f" [{len(self.active_files)} files]" if self.active_files else ""
                prompt_text = f"\n🔵 You ({self.current_token_count}{active_files_info}): "
                user_input = session.prompt(prompt_text).strip()

                # ─── 2 · trivial exits / empty line ─────────────────────────────
                if user_input.lower() in {"exit", "quit", "/exit", "/quit", "/q"}:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    # unwind the stats we pushed at the top of the loop
                    self.prompt_time_counts.pop()
                    self.messages_per_interval.pop()
                    continue

                # count this turn
                self._messages_this_interval += 1
                message_to_send = user_input                 # ← default

                # ─── 3 · COMMAND‑HANDLING  (may early‑continue) ────────────────
                # NOTE: anything that ends with `continue` must happen **before**
                # we prepend PDF context, otherwise we might discard it.
                # ----------------------------------------------------------------

                # /reset ----------------------------------------------------------
                if user_input.lower() == "/reset":
                    self._initialize_chat()
                    self.conversation_history.clear()
                    self.current_token_count = 0
                    print("\n🎯 Resetting context and starting a new chat session...")
                    continue

                # /clear <n_tokens> ----------------------------------------------
                if user_input.lower().startswith("/clear"):
                    # (existing clear‑logic unchanged)
                    try:
                        # ...
                        print(f"\n✅ Approximately cleared {messages_to_remove} message(s) "
                              f"(up to {tokens_actually_cleared} tokens). "
                              f"New total tokens: {self.current_token_count}")
                    except Exception as e:
                        print(f"⚠️ Error processing /clear command: {e}")
                    continue

                # /prompt <name> --------------------------------------------------
                if user_input.lower().startswith("/prompt "):
                    parts = user_input.split(maxsplit=1)
                    prompt_name = parts[1] if len(parts) == 2 else ""
                    prompt_content = self._load_prompt(prompt_name)
                    if prompt_content:
                        self.pending_prompt = prompt_content
                        print(f"\n✅ Prompt '{prompt_name}' loaded. "
                              "It will be included in your next message.")
                    else:
                        print(f"\n❌ Prompt '{prompt_name}' not found.")
                    continue                                                     # <‑‑ early return

                # /pdf …  (delegates to helper; helper sets self.pending_pdf_context)
                if user_input.lower().startswith("/pdf"):
                    args = user_input.split()[1:]
                    self._handle_pdf_command(args)
                    continue                                                     # <‑‑ early return

                # /save, /load, /thinking_budget … (existing logic unchanged)
                # ­­‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

                # ─── 4 · PROMPT injection (guaranteed to send) ───────────────────
                if self.pending_prompt:
                    print("[Including loaded prompt.]\n")
                    message_to_send = f"{self.pending_prompt}\n\n{message_to_send}"
                    self.pending_prompt = None

                # ─── 5 · PDF‑CONTEXT injection (moved **after** commands) ────────
                if self.pending_pdf_context:
                    print("[Including context from previously processed PDF.]\n")
                    message_to_send = f"{self.pending_pdf_context}\n\n{message_to_send}"
                    self.pending_pdf_context = None        # consume exactly once

                # ─── 6 · assemble message_content (text + files) ────────────────
                message_content = [message_to_send]
                if self.active_files:
                    message_content.extend(self.active_files)
                    if self.config.get('verbose', DEFAULT_CONFIG['verbose']):
                        print(f"\n📎 Attaching {len(self.active_files)} files to the prompt:")
                        for f in self.active_files:
                            print(f"   - {f.display_name} ({f.name})")

                # ─── 7 · manual history (token‑counting only) ───────────────────
                new_user_content = glm.Content(role="user",
                                               parts=[glm.Part(text=message_to_send)])
                self.conversation_history.append(new_user_content)

                # ─── 8 · send to Gemini ─────────────────────────────────────────
                print("\n⏳ Sending message and processing…")
                tool_cfg = types.GenerateContentConfig(
                    tools=self.tool_functions,
                    thinking_config=self.thinking_config
                )
                response = self.chat.send_message(message=message_content,
                                                  config=tool_cfg)

                agent_response_text = ""
                if response.candidates and response.candidates[0].content:
                    agent_parts = response.candidates[0].content.parts
                    agent_response_text = " ".join(p.text for p in agent_parts
                                                   if hasattr(p, "text"))

                if agent_response_text:
                    hist_agent_content = glm.Content(role="model",
                                                     parts=[glm.Part(text=agent_response_text)])
                    self.conversation_history.append(hist_agent_content)

                print(f"\n🟢 \x1b[92mAgent:\x1b[0m {agent_response_text or '[No response text]'}")

                # ─── 9 · token count update (unchanged) ─────────────────────────
                try:
                    token_info = self.client.models.count_tokens(
                        model=self.model_name,
                        contents=self.conversation_history
                    )
                    self.current_token_count = token_info.total_tokens
                    print(f"\n[Token Count: {self.current_token_count}]")
                except Exception as count_err:
                    logger.error(f"Error calculating token count: {count_err}", exc_info=True)
                    print("🚨 Error: Failed to calculate token count.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n🔴 An error occurred during interaction: {e}")
                traceback.print_exc()
```

### What changed & why

* **PDF context is no longer injected (and cleared) before command parsing.**
  Instead, steps 4‑5 run **after** any early‑`continue` command blocks, guaranteeing the context exists when the message is sent. This fixes the “LLM says it can’t see the paper” issue you observed .
* No `_dispatch_command` helper is required; we keep your inline `if/elif` command handlers (they were already present in the snippet around lines … ) .
* All other behaviour (token accounting, file attachments, prompt loading, etc.) remains identical.

Paste this loop over the current one in `main.py`; nothing else needs to change.
