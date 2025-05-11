genai.caches module
class genai.caches.AsyncCaches(api_client_)
Bases: BaseModule

async create(*, model, config=None)
Creates a cached contents resource.

Usage:

contents = ... // Initialize the content to cache.
response = await client.aio.caches.create(
    model= ... // The publisher model id
    contents=contents,
    config={
        'display_name': 'test cache',
        'system_instruction': 'What is the sum of the two pdfs?',
        'ttl': '86400s',
    },
)
Return type:
CachedContent

async delete(*, name, config=None)
Deletes cached content.

Usage:

await client.aio.caches.delete(name= ... ) // The server-generated
resource name.
Return type:
DeleteCachedContentResponse

async get(*, name, config=None)
Gets cached content configurations.

await client.aio.caches.get(name= ... ) // The server-generated resource
name.
Return type:
CachedContent

async list(*, config=None)
Return type:
AsyncPager[CachedContent]

async update(*, name, config=None)
Updates cached content configurations.

response = await client.aio.caches.update(
    name= ... // The server-generated resource name.
    config={
        'ttl': '7600s',
    },
)
Return type:
CachedContent

class genai.caches.Caches(api_client_)
Bases: BaseModule

create(*, model, config=None)
Creates a cached contents resource.

Usage:

contents = ... // Initialize the content to cache.
response = client.caches.create(
    model= ... // The publisher model id
    contents=contents,
    config={
        'display_name': 'test cache',
        'system_instruction': 'What is the sum of the two pdfs?',
        'ttl': '86400s',
    },
)
Return type:
CachedContent

delete(*, name, config=None)
Deletes cached content.

Usage:

client.caches.delete(name= ... ) // The server-generated resource name.
Return type:
DeleteCachedContentResponse

get(*, name, config=None)
Gets cached content configurations.

client.caches.get(name= ... ) // The server-generated resource name.
Return type:
CachedContent

list(*, config=None)
Return type:
Pager[CachedContent]

update(*, name, config=None)
Updates cached content configurations.

response = client.caches.update(
    name= ... // The server-generated resource name.
    config={
        'ttl': '7600s',
    },
)
Return type:
CachedContent