#!/usr/bin/env python
# coding: utf-8

from google import genai
from google.genai import types
import pathlib
import httpx
import os
from pypdf import PdfReader, PdfWriter
from io import BytesIO

# load GEMINI_API_KEY from .env file
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


get_ipython().system('ls ../papers/reasoning/202501/')


client = genai.Client(api_key=GEMINI_API_KEY)

# doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"

# # Retrieve and encode the PDF byte
# filepath = pathlib.Path('file.pdf')
# filepath.write_bytes(httpx.get(doc_url).content)


# 1. read the original PDF
reader = PdfReader("../papers/reasoning/202501/DeepSeek_R1.pdf")

# 2. pick the pages you want (zero-based indices)
pages_to_send = [4, 5]   # e.g. pages 1, 2 and 3

writer = PdfWriter()
for idx in pages_to_send:
    writer.add_page(reader.pages[idx])

# 3. write them to a bytes buffer
buf = BytesIO()
writer.write(buf)
buf.seek(0)
subset_pdf_bytes = buf.read()

# 4. call Gemini with only that subset
client = genai.Client(api_key=GEMINI_API_KEY)
prompt = "Extract the full text of this document with figure/table descriptions. Render the text in markdown format, makeing sure to use LaTeX for equations. Also render any mathematical variables or expressions that are present in the text using inline LaTeX. Pay close attention to proper LaTeX formatting including bracket nesting, and understanding the difference between what is mathematical notation, and what is a text string within an equation. Make sure the latex snippets are properly enclosed using dollar signs so that both the inline LaTex and standalone equations are rendered correctly in markdown. Anything enclosed with $$ is a standalone equation, and anything enclosed with $ is an inline equation."

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
      types.Part.from_bytes(data=subset_pdf_bytes,
                            mime_type="application/pdf"),
      prompt
    ]
)

print(response.text)


filepath = pathlib.Path('../papers/reasoning/202501/DeepSeek_R1.pdf')
prompt = "Extract the full text of the document including detailed descriptions of the figures and tables."
response = client.models.generate_content(
  model="gemini-2.0-flash",
  contents=[
      types.Part.from_bytes(
        data=filepath.read_bytes(),
        mime_type='application/pdf',
      ),
      prompt])
print(response.text)








get_ipython().system('jupyter nbconvert gemini_pdf.ipynb    --to python    --TemplateExporter.exclude_output=True    --TemplateExporter.exclude_input_prompt=True')




