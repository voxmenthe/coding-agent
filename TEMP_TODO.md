We need to add a `/history` command that prints the entirety of the current conversation context to the cli for the user to inspect. We also need an `edit_history` command that allows us to edit the current history.

We also need a `/sql` command that allows us to run arbitrary sql queries against the paper database.

Also if you hit enter twice it should send the current conversation history to the agent, even if the user has not entered anything on the current line.

