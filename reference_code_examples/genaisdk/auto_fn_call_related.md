pydantic model genai.types.GenerateContentConfig
Bases: BaseModel

Optional model configuration parameters.

For more information, see Content generation parameters.

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

Show JSON schema
Fields:
audio_timestamp (bool | None)

automatic_function_calling (genai.types.AutomaticFunctionCallingConfig | None)

cached_content (str | None)

candidate_count (int | None)

frequency_penalty (float | None)

http_options (genai.types.HttpOptions | None)

labels (dict[str, str] | None)

logprobs (int | None)

max_output_tokens (int | None)

media_resolution (genai.types.MediaResolution | None)

model_selection_config (genai.types.ModelSelectionConfig | None)

presence_penalty (float | None)

response_logprobs (bool | None)

response_mime_type (str | None)

response_modalities (list[str] | None)

response_schema (dict[Any, Any] | type | genai.types.Schema | types.GenericAlias | types.UnionType | _UnionGenericAlias | None)

routing_config (genai.types.GenerationConfigRoutingConfig | None)

safety_settings (list[genai.types.SafetySetting] | None)

seed (int | None)

speech_config (genai.types.SpeechConfig | str | None)

stop_sequences (list[str] | None)

system_instruction (genai.types.Content | list[genai.types.File | genai.types.Part | PIL.Image.Image | str] | genai.types.File | genai.types.Part | PIL.Image.Image | str | None)

temperature (float | None)

thinking_config (genai.types.ThinkingConfig | None)

tool_config (genai.types.ToolConfig | None)

tools (list[genai.types.Tool | Callable[[...], Any]] | None)

top_k (float | None)

top_p (float | None)

Validators:
_convert_literal_to_enum » response_schema

field audio_timestamp: Optional[bool] = None (alias 'audioTimestamp')
If enabled, audio timestamp will be included in the request to the model.

field automatic_function_calling: Optional[AutomaticFunctionCallingConfig] = None (alias 'automaticFunctionCalling')
The configuration for automatic function calling.

field cached_content: Optional[str] = None (alias 'cachedContent')
Resource name of a context cache that can be used in subsequent requests.

field candidate_count: Optional[int] = None (alias 'candidateCount')
Number of response variations to return.

field frequency_penalty: Optional[float] = None (alias 'frequencyPenalty')
Positive values penalize tokens that repeatedly appear in the generated text, increasing the probability of generating more diverse content.

field http_options: Optional[HttpOptions] = None (alias 'httpOptions')
Used to override HTTP request options.

field labels: Optional[dict[str, str]] = None
Labels with user-defined metadata to break down billed charges.

field logprobs: Optional[int] = None
Number of top candidate tokens to return the log probabilities for at each generation step.

field max_output_tokens: Optional[int] = None (alias 'maxOutputTokens')
Maximum number of tokens that can be generated in the response.

field media_resolution: Optional[MediaResolution] = None (alias 'mediaResolution')
If specified, the media resolution specified will be used.

field model_selection_config: Optional[ModelSelectionConfig] = None (alias 'modelSelectionConfig')
Configuration for model selection.

field presence_penalty: Optional[float] = None (alias 'presencePenalty')
Positive values penalize tokens that already appear in the generated text, increasing the probability of generating more diverse content.

field response_logprobs: Optional[bool] = None (alias 'responseLogprobs')
Whether to return the log probabilities of the tokens that were chosen by the model at each step.

field response_mime_type: Optional[str] = None (alias 'responseMimeType')
Output response mimetype of the generated candidate text. Supported mimetype:

text/plain: (default) Text output.

application/json: JSON response in the candidates.

The model needs to be prompted to output the appropriate response type, otherwise the behavior is undefined. This is a preview feature.

field response_modalities: Optional[list[str]] = None (alias 'responseModalities')
The requested modalities of the response. Represents the set of modalities that the model can return.

field response_schema: Union[dict[Any, Any], type, Schema, GenericAlias, , _UnionGenericAlias, None] = None (alias 'responseSchema')
The Schema object allows the definition of input and output data types. These types can be objects, but also primitives and arrays. Represents a select subset of an [OpenAPI 3.0 schema object](https://spec.openapis.org/oas/v3.0.3#schema). If set, a compatible response_mime_type must also be set. Compatible mimetypes: application/json: Schema for JSON response.

Validated by:
_convert_literal_to_enum

field routing_config: Optional[GenerationConfigRoutingConfig] = None (alias 'routingConfig')
Configuration for model router requests.

field safety_settings: Optional[list[SafetySetting]] = None (alias 'safetySettings')
Safety settings in the request to block unsafe content in the response.

field seed: Optional[int] = None
When seed is fixed to a specific number, the model makes a best effort to provide the same response for repeated requests. By default, a random number is used.

field speech_config: Union[SpeechConfig, str, None] = None (alias 'speechConfig')
The speech generation configuration.

field stop_sequences: Optional[list[str]] = None (alias 'stopSequences')
List of strings that tells the model to stop generating text if one of the strings is encountered in the response.

field system_instruction: Union[Content, list[Union[File, Part, Image, str]], File, Part, Image, str, None] = None (alias 'systemInstruction')
Instructions for the model to steer it toward better performance. For example, “Answer as concisely as possible” or “Don’t use technical terms in your response”.

field temperature: Optional[float] = None
Value that controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results.

field thinking_config: Optional[ThinkingConfig] = None (alias 'thinkingConfig')
The thinking features configuration.

field tool_config: Optional[ToolConfig] = None (alias 'toolConfig')
Associates model output to a specific function call.

field tools: Optional[list[Union[Tool, Callable[..., Any]]]] = None
Code that enables the system to interact with external systems to perform an action outside of the knowledge and scope of the model.

field top_k: Optional[float] = None (alias 'topK')
For each token selection step, the top_k tokens with the highest probabilities are sampled. Then tokens are further filtered based on top_p with the final token selected using temperature sampling. Use a lower number for less random responses and a higher number for more random responses.

field top_p: Optional[float] = None (alias 'topP')
Tokens are selected from the most to least probable until the sum of their probabilities equals this value. Use a lower value for less random responses and a higher value for more random responses.

class genai.types.GenerateContentConfigDict
Bases: TypedDict

Optional model configuration parameters.

For more information, see Content generation parameters.

audio_timestamp: Optional[bool]
If enabled, audio timestamp will be included in the request to the model.

automatic_function_calling: Optional[AutomaticFunctionCallingConfigDict]
The configuration for automatic function calling.

cached_content: Optional[str]
Resource name of a context cache that can be used in subsequent requests.

candidate_count: Optional[int]
Number of response variations to return.

frequency_penalty: Optional[float]
Positive values penalize tokens that repeatedly appear in the generated text, increasing the probability of generating more diverse content.

http_options: Optional[HttpOptionsDict]
Used to override HTTP request options.

labels: Optional[dict[str, str]]
Labels with user-defined metadata to break down billed charges.

logprobs: Optional[int]
Number of top candidate tokens to return the log probabilities for at each generation step.

max_output_tokens: Optional[int]
Maximum number of tokens that can be generated in the response.

media_resolution: Optional[MediaResolution]
If specified, the media resolution specified will be used.

model_selection_config: Optional[ModelSelectionConfigDict]
Configuration for model selection.

presence_penalty: Optional[float]
Positive values penalize tokens that already appear in the generated text, increasing the probability of generating more diverse content.

response_logprobs: Optional[bool]
Whether to return the log probabilities of the tokens that were chosen by the model at each step.

response_mime_type: Optional[str]
Output response mimetype of the generated candidate text. Supported mimetype:

text/plain: (default) Text output.

application/json: JSON response in the candidates.

The model needs to be prompted to output the appropriate response type, otherwise the behavior is undefined. This is a preview feature.

response_modalities: Optional[list[str]]
The requested modalities of the response. Represents the set of modalities that the model can return.

response_schema: Union[dict[Any, Any], type, Schema, GenericAlias, , _UnionGenericAlias, SchemaDict, None]
The Schema object allows the definition of input and output data types. These types can be objects, but also primitives and arrays. Represents a select subset of an [OpenAPI 3.0 schema object](https://spec.openapis.org/oas/v3.0.3#schema). If set, a compatible response_mime_type must also be set. Compatible mimetypes: application/json: Schema for JSON response.

routing_config: Optional[GenerationConfigRoutingConfigDict]
Configuration for model router requests.

safety_settings: Optional[list[SafetySettingDict]]
Safety settings in the request to block unsafe content in the response.

seed: Optional[int]
When seed is fixed to a specific number, the model makes a best effort to provide the same response for repeated requests. By default, a random number is used.

speech_config: Union[SpeechConfig, str, SpeechConfigDict, None]
The speech generation configuration.

stop_sequences: Optional[list[str]]
List of strings that tells the model to stop generating text if one of the strings is encountered in the response.

system_instruction: Union[Content, list[Union[File, Part, Image, str]], File, Part, Image, str, ContentDict, None]
Instructions for the model to steer it toward better performance. For example, “Answer as concisely as possible” or “Don’t use technical terms in your response”.

temperature: Optional[float]
Value that controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results.

thinking_config: Optional[ThinkingConfigDict]
The thinking features configuration.

tool_config: Optional[ToolConfigDict]
Associates model output to a specific function call.

tools: Optional[list[Union[ToolDict, Callable[..., Any]]]]
Code that enables the system to interact with external systems to perform an action outside of the knowledge and scope of the model.

top_k: Optional[float]
For each token selection step, the top_k tokens with the highest probabilities are sampled. Then tokens are further filtered based on top_p with the final token selected using temperature sampling. Use a lower number for less random responses and a higher number for more random responses.

top_p: Optional[float]
Tokens are selected from the most to least probable until the sum of their probabilities equals this value. Use a lower value for less random responses and a higher value for more random responses.

pydantic model genai.types.GenerateContentResponse
Bases: BaseModel

Response message for PredictionService.GenerateContent.

Create a new model by parsing and validating input data from keyword arguments.

Raises [ValidationError][pydantic_core.ValidationError] if the input data cannot be validated to form a valid model.

self is explicitly positional-only to allow self as a field name.

Show JSON schema
Fields:
automatic_function_calling_history (list[genai.types.Content] | None)

candidates (list[genai.types.Candidate] | None)

create_time (datetime.datetime | None)

model_version (str | None)

parsed (pydantic.main.BaseModel | dict[Any, Any] | enum.Enum | None)

prompt_feedback (genai.types.GenerateContentResponsePromptFeedback | None)

response_id (str | None)

usage_metadata (genai.types.GenerateContentResponseUsageMetadata | None)

field automatic_function_calling_history: Optional[list[Content]] = None (alias 'automaticFunctionCallingHistory')
field candidates: Optional[list[Candidate]] = None
Response variations returned by the model.

field create_time: Optional[datetime] = None (alias 'createTime')
Timestamp when the request is made to the server.

field model_version: Optional[str] = None (alias 'modelVersion')
Output only. The model version used to generate the response.

field parsed: Union[BaseModel, dict[Any, Any], Enum, None] = None
First candidate from the parsed response if response_schema is provided. Not available for streaming.

field prompt_feedback: Optional[GenerateContentResponsePromptFeedback] = None (alias 'promptFeedback')
Output only. Content filter results for a prompt sent in the request. Note: Sent only in the first stream chunk. Only happens when no candidates were generated due to content violations.

field response_id: Optional[str] = None (alias 'responseId')
Identifier for each response.

field usage_metadata: Optional[GenerateContentResponseUsageMetadata] = None (alias 'usageMetadata')
Usage metadata about the response(s).

property code_execution_result: str | None
Returns the code execution result in the response.

property executable_code: str | None
Returns the executable code in the response.

property function_calls: list[FunctionCall] | None
Returns the list of function calls in the response.

property text: str | None
Returns the concatenation of all text parts in the response.

class genai.types.GenerateContentResponseDict
Bases: TypedDict

Response message for PredictionService.GenerateContent.

candidates: Optional[list[CandidateDict]]
Response variations returned by the model.

create_time: Optional[datetime]
Timestamp when the request is made to the server.

model_version: Optional[str]
Output only. The model version used to generate the response.

prompt_feedback: Optional[GenerateContentResponsePromptFeedbackDict]
Sent only in the first stream chunk. Only happens when no candidates were generated due to content violations.

Type:
Output only. Content filter results for a prompt sent in the request. Note

response_id: Optional[str]
Identifier for each response.

usage_metadata: Optional[GenerateContentResponseUsageMetadataDict]
Usage metadata about the response(s).