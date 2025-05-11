genai.files module
class genai.files.AsyncFiles(api_client_)
Bases: BaseModule

async delete(*, name, config=None)
Deletes a remotely stored file.

Return type:
DeleteFileResponse

Parameters:
name (str) – The name identifier for the file to delete.

config (DeleteFileConfig) – Optional, configuration for the delete method.

Returns:
The response for the delete method

Return type:
DeleteFileResponse

Usage:

await client.aio.files.delete(name='files/...')
async download(*, file, config=None)
Downloads a file’s data from the file service.

The Vertex-AI implementation of the API foes not include the file service.

Files created by upload can’t be downloaded. You can tell which files are downloadable by checking the download_uri property.

Return type:
bytes

Parameters:
File (str) – A file name, uri, or file object. Identifying which file to download.

config (DownloadFileConfigOrDict) – Optional, configuration for the get method.

Returns:
The file data as bytes.

Return type:
File

Usage:

for file client.files.list():
  if file.download_uri is not None:
    break
else:
  raise ValueError('No files found with a `download_uri`.')
data = client.files.download(file=file)
# data = client.files.download(file=file.name)
# data = client.files.download(file=file.uri)
async get(*, name, config=None)
Retrieves the file information from the service.

Return type:
File

Parameters:
name (str) – The name identifier for the file to retrieve.

config (GetFileConfig) – Optional, configuration for the get method.

Returns:
The file information.

Return type:
File

Usage:

file = await client.aio.files.get(name='files/...')
print(file.uri)
async list(*, config=None)
Return type:
AsyncPager[File]

async upload(*, file, config=None)
Calls the API to upload a file asynchronously using a supported file service.

Return type:
File

Parameters:
file – A path to the file or an IOBase object to be uploaded. If it’s an IOBase object, it must be opened in blocking (the default) mode and binary mode. In other words, do not use non-blocking mode or text mode. The given stream must be seekable, that is, it must be able to call seek() on ‘path’.

config – Optional parameters to set diplay_name, mime_type, and name.

class genai.files.Files(api_client_)
Bases: BaseModule

delete(*, name, config=None)
Deletes a remotely stored file.

Return type:
DeleteFileResponse

Parameters:
name (str) – The name identifier for the file to delete.

config (DeleteFileConfig) – Optional, configuration for the delete method.

Returns:
The response for the delete method

Return type:
DeleteFileResponse

Usage:

client.files.delete(name='files/...')
download(*, file, config=None)
Downloads a file’s data from storage.

Files created by upload can’t be downloaded. You can tell which files are downloadable by checking the source or download_uri property.

Note: This method returns the data as bytes. For Video and GeneratedVideo objects there is an additional side effect, that it also sets the video_bytes property on the Video object.

Return type:
bytes

Parameters:
file (str) – A file name, uri, or file object. Identifying which file to download.

config (DownloadFileConfigOrDict) – Optional, configuration for the get method.

Returns:
The file data as bytes.

Return type:
File

Usage:

for file client.files.list():
  if file.download_uri is not None:
    break
else:
  raise ValueError('No files found with a `download_uri`.')
data = client.files.download(file=file)
# data = client.files.download(file=file.name)
# data = client.files.download(file=file.download_uri)

video = types.Video(uri=file.uri)
video_bytes = client.files.download(file=video)
video.video_bytes
get(*, name, config=None)
Retrieves the file information from the service.

Return type:
File

Parameters:
name (str) – The name identifier for the file to retrieve.

config (GetFileConfig) – Optional, configuration for the get method.

Returns:
The file information.

Return type:
File

Usage:

file = client.files.get(name='files/...')
print(file.uri)
list(*, config=None)
Return type:
Pager[File]

upload(*, file, config=None)
Calls the API to upload a file using a supported file service.

Return type:
File

Parameters:
file – A path to the file or an IOBase object to be uploaded. If it’s an IOBase object, it must be opened in blocking (the default) mode and binary mode. In other words, do not use non-blocking mode or text mode. The given stream must be seekable, that is, it must be able to call seek() on ‘path’.

config – Optional parameters to set diplay_name, mime_type, and name.

