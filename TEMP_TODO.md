1. We need to add a `/history` command that prints the entirety of the current conversation context to the cli for the user to inspect. We also need an `edit_history` command that allows us to edit the current history.

2. We also need a `/sql` command that allows us to run arbitrary sql queries against the paper database.

3. Also if you hit enter twice it should send the current conversation history to the agent, even if the user has not entered anything on the current line.

4. Modify the /prompt command to allow for an additional (optional) typed prompt after the prompt name, so that you can do a /prompt <prompt_name> <prompt_text> command to load a prompt from the prompts directory and append the typed prompt text to it in one go.

5. When loading a pdf (identified by filename), if it has been loaded before, we should use the cached version from the db instead of reprocessing it.
