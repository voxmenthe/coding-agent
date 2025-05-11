genai.chats module
class genai.chats.AsyncChat(*, modules, model, config=None, history)
Bases: _BaseChat

Async chat session.

async send_message(message, config=None)
Sends the conversation history with the additional message and returns model’s response.

Return type:
GenerateContentResponse

Parameters:
message – The message to send to the model.

config – Optional config to override the default Chat config for this request.

Returns:
The model’s response.

Usage:

chat = client.aio.chats.create(model='gemini-2.0-flash')
response = await chat.send_message('tell me a story')
async send_message_stream(message, config=None)
Sends the conversation history with the additional message and yields the model’s response in chunks.

Return type:
AsyncIterator[GenerateContentResponse]

Parameters:
message – The message to send to the model.

config – Optional config to override the default Chat config for this request.

Yields:
The model’s response in chunks.

Usage:

class genai.chats.AsyncChats(modules)
Bases: object

A util class to create async chat sessions.

create(*, model, config=None, history=None)
Creates a new chat session.

Return type:
AsyncChat

Parameters:
model – The model to use for the chat.

config – The configuration to use for the generate content request.

history – The history to use for the chat.

Returns:
A new chat session.

class genai.chats.Chat(*, modules, model, config=None, history)
Bases: _BaseChat

Chat session.

send_message(message, config=None)
Sends the conversation history with the additional message and returns the model’s response.

Return type:
GenerateContentResponse

Parameters:
message – The message to send to the model.

config – Optional config to override the default Chat config for this request.

Returns:
The model’s response.

Usage:

chat = client.chats.create(model='gemini-2.0-flash')
response = chat.send_message('tell me a story')
send_message_stream(message, config=None)
Sends the conversation history with the additional message and yields the model’s response in chunks.

Return type:
Iterator[GenerateContentResponse]

Parameters:
message – The message to send to the model.

config – Optional config to override the default Chat config for this request.

Yields:
The model’s response in chunks.

Usage:

chat = client.chats.create(model='gemini-2.0-flash')
for chunk in chat.send_message_stream('tell me a story'):
  print(chunk.text)
class genai.chats.Chats(modules)
Bases: object

A util class to create chat sessions.

create(*, model, config=None, history=None)
Creates a new chat session.

Return type:
Chat

Parameters:
model – The model to use for the chat.

config – The configuration to use for the generate content request.

history – The history to use for the chat.

Returns:
A new chat session.