
1. We also need an `edit_history` command that allows us to edit the current history.

2. We also need a `/sql` command that allows us to run arbitrary sql queries against the paper database.

3. Can we improve the CLI functionality in this codebase to allow for multi-line edits and easier cursor-based editing in general?

3. Also if you hit enter twice it should send the current conversation history to the agent, even if the user has not entered anything on the current line.

4. Add a /model command that allows us to switch between models. (and maybe add a couple more providers as well - like Qwen and DeepSeek)

5. This logging is way too verbose. We need to be able to turn it off and on with a command like `/toggle_verbose` or something like that (and it tells you the current setting).

Example logs:
🔵 You (0): INFO     [google_genai.models] AFC remote call 1 is done.
INFO     [src.tools] Successfully extracted text from DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf.
INFO     [src.tools] Deleting Gemini file files/djgbm8q30v8j...
INFO     [src.tools] Successfully deleted Gemini file files/djgbm8q30v8j.
INFO     [src.main] Finalize Thread Paper ID 3: Successfully extracted text (39791 chars).
INFO     [src.main] Finalize Thread Paper ID 3: Saving extracted text to /Volumes/bdrive/repos/coding-agent/processed_papers/blobs/paper_3_text.txt.
INFO     [src.tools] Successfully saved text blob to /Volumes/bdrive/repos/coding-agent/processed_papers/blobs/paper_3_text.txt
INFO     [src.main] Finalize Thread Paper ID 3: Successfully saved text blob. Updating DB.
INFO     [src.database] Updated field 'blob_path' for paper ID 3.
INFO     [src.database] Updated field 'genai_file_uri' for paper ID 3.
INFO     [src.database] Updated field 'processed_timestamp' for paper ID 3.
INFO     [src.main] Finalize Thread Paper ID 3: Stored context (39900 chars). Updating status.
INFO     [src.database] Updated field 'status' for paper ID 3.
INFO     [src.main] Finalize Thread Paper ID 3: Finalization complete for 'DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf'.
INFO     [src.main] Finalize Thread Paper ID 3: Closing thread-local DB connection.
INFO     [src.database] Database connection closed.
INFO     [root] Task b65ff857-3f8d-4aa8-be41-5f42fd066815: Successfully processed and finalized DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf

✅ Task 'PDF-DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf' (ID: b65ff857-3f8d-4aa8-be41-5f42fd066815) completed successfully. Result: Successfully processed DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf
INFO     [root] Task 'PDF-DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf' (ID: b65ff857-3f8d-4aa8-be41-5f42fd066815) completed successfully. Result: Successfully processed DAPO-AnOpen-SourceLLMReinforcementLearningSystematScale2503.14476v1.pdf

6. Should be able to, for example, give a tool to "pull" the latest arxiv papers on a given topic given a time period and save them to disk and also, if asked, load them into the conversation.

7. Need to test if I can get it to pull a specific paper from arxiv given a rough description of the paper and/or name.

8. Launch agentic workflows. (Does this need a separate tool call that instantiates a new LLM loop? Yes probably.)

9. Implement DSPY or some other kind of structured output for the LLM to use when searching for arxiv papers - give it a process for discovering which papers I'd find most relevant to a given query.

Example query:
Find all of the arxiv papers in the CS or Stats categories published between Apr 1 2025 and May 12
2025 having to do with either LLM reasoning, reasoning models, LLM RL rewards, RL for LLMs, test time scaling,
GRPO, or verifiers. Use a wide variety of keyword including synonyms of all of the above to make sure we captur
e as many of them as possible (for example LLM can also be known as "language model" and other likely search te
rms that stem from the context described above might include many others that can be included - as an example "
RLHF" is one - I'm sure you can think of others. Display 200 results