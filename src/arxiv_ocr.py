# arxiv_ocr.py
# Usage Examples:
#     python arxiv_ocr.py https://arxiv.org/abs/1706.03762 --api-key YOUR_API_KEY
#     python arxiv_ocr.py --file-path path/to/paper.pdf --pages 5 --html

import os
import json
import base64
import re
import requests
import tempfile
from pathlib import Path
import unicodedata
from urllib.parse import urlparse
import click
import markdown
from bs4 import BeautifulSoup
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from creds import all_creds

MISTRAL_API_KEY = all_creds['MISTRAL_API_KEY']
os.environ['MISTRAL_API_KEY'] = MISTRAL_API_KEY

def get_arxiv_pdf(arxiv_id):
    """
    Download the PDF for an arXiv ID
    
    Args:
        arxiv_id (str): arXiv ID
    
    Returns:
        bytes: PDF content
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(pdf_url, headers=headers)
    
    if response.status_code != 200:
        raise ValueError(f"Could not download PDF. Status code: {response.status_code}")
    
    return response.content

def extract_arxiv_id(url):
    """
    Extract arXiv ID from an arXiv URL (abs, PDF, or HTML)
    
    Args:
        url (str): arXiv URL
    
    Returns:
        str: arXiv ID
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if it's an arXiv URL
    if 'arxiv.org' not in parsed_url.netloc:
        raise ValueError("Not an arXiv URL")
    
    # Extract the arXiv ID
    path_parts = parsed_url.path.strip('/').split('/')
    
    # Handle different URL formats
    arxiv_id = None
    
    if 'abs' in path_parts:
        # Format: arxiv.org/abs/1234.56789
        idx = path_parts.index('abs')
        if idx + 1 < len(path_parts):
            arxiv_id = path_parts[idx + 1]
    elif 'pdf' in path_parts:
        # Format: arxiv.org/pdf/1234.56789.pdf
        idx = path_parts.index('pdf')
        if idx + 1 < len(path_parts):
            arxiv_id = path_parts[idx + 1].replace('.pdf', '')
    elif 'html' in path_parts:
        # Format: arxiv.org/html/1234.56789
        idx = path_parts.index('html')
        if idx + 1 < len(path_parts):
            arxiv_id = path_parts[idx + 1]
    else:
        # Try to find the ID in the last part of the path
        last_part = path_parts[-1]
        if re.match(r'\d+\.\d+', last_part):
            arxiv_id = last_part
    
    if not arxiv_id:
        raise ValueError("Could not extract arXiv ID from URL")
    
    return arxiv_id


def get_arxiv_bibtex(arxiv_id):
    """
    Download the BibTeX citation for an arXiv ID
    
    Args:
        arxiv_id (str): arXiv ID
    
    Returns:
        str: BibTeX citation text or None if not available
    """
    # arXiv's BibTeX endpoint
    bibtex_url = f"https://arxiv.org/bibtex/{arxiv_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(bibtex_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to get BibTeX: HTTP {response.status_code}")
            return None
        
        # Extract BibTeX content from the response
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # First try to find the textarea that usually contains the BibTeX
        textarea = soup.find('textarea')
        if textarea:
            return textarea.get_text().strip()
        
        # If no textarea, try to extract pre-formatted text
        pre = soup.find('pre')
        if pre:
            return pre.get_text().strip()
        
        # Last resort: generate a basic BibTeX entry ourselves
        print("Could not find BibTeX on the page, generating a basic entry")
        
        # We'll need the title and authors
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
        abs_response = requests.get(abs_url, headers=headers)
        
        if abs_response.status_code == 200:
            abs_soup = BeautifulSoup(abs_response.text, 'html.parser')
            
            # Extract title
            title_elem = abs_soup.find('h1', class_='title')
            title = title_elem.get_text().replace('Title:', '').strip() if title_elem else "Unknown Title"
            
            # Extract authors
            authors_elem = abs_soup.find('div', class_='authors')
            authors = authors_elem.get_text().replace('Authors:', '').strip() if authors_elem else "Unknown Authors"
            
            # Extract year
            year = "2023"  # Default to current year if we can't find it
            date_elem = abs_soup.find('div', class_='dateline')
            if date_elem:
                date_match = re.search(r'\b(19|20)\d{2}\b', date_elem.get_text())
                if date_match:
                    year = date_match.group(0)
            
            # Generate a simple BibTeX entry
            bibtex = f"""@article{{{arxiv_id},
  title = {{{title}}},
  author = {{{authors}}},
  journal = {{arXiv preprint arXiv:{arxiv_id}}},
  year = {{{year}}},
  url = {{https://arxiv.org/abs/{arxiv_id}}},
}}"""
            return bibtex
    
    except Exception as e:
        print(f"Error fetching BibTeX: {str(e)}")
        return None
    
    return None


def get_paper_metadata(arxiv_id):
    """
    Get the paper title and submission date from arXiv abstract page
    
    Args:
        arxiv_id (str): arXiv ID
    
    Returns:
        tuple: (Paper title, Submission date)
    """
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(abs_url, headers=headers)
    
    if response.status_code != 200:
        return "unknown-paper", None
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract title
    title = "unknown-paper"
    title_element = soup.find('h1', class_='title')
    if title_element:
        title = title_element.get_text().replace('Title:', '').strip()
    
    # Extract submission date
    submission_date = None
    submission_element = soup.find('div', class_='submission-history')
    if submission_element:
        # Look for the first submission date
        submission_text = submission_element.get_text()
        match = re.search(r'\[v1\]\s+(.+?)\s+\(', submission_text)
        if not match:
            # Try alternative pattern without version
            match = re.search(r'Submitted\s+(.+?)\s+\(', submission_text)
        
        if match:
            submission_date = match.group(1).strip()
    
    return title, submission_date


def sanitize_filename(title):
    """
    Convert title to a clean filename format
    
    Args:
        title (str): Paper title
    
    Returns:
        str: Sanitized filename
    """
    # Convert to lowercase and replace spaces with hyphens
    filename = title.lower()
    
    # Remove accents
    filename = ''.join(c for c in unicodedata.normalize('NFKD', filename)
                     if not unicodedata.combining(c))
    
    # Replace non-alphanumeric characters with hyphens
    filename = re.sub(r'[^a-z0-9]', '-', filename)
    
    # Replace multiple hyphens with a single hyphen
    filename = re.sub(r'-+', '-', filename)
    
    # Remove leading and trailing hyphens
    filename = filename.strip('-')
    
    return filename


@click.command()
@click.argument("arxiv_url", required=False)
@click.option(
    "--file-path",
    "-f",
    help="Path to a local PDF file to process instead of downloading from arXiv.",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-file",
    "-o",
    help="Output filename. If not provided, defaults to {sanitized_title}.md or {sanitized_title}.html if --html is used.",
    type=str,
)
@click.option(
    "--api-key",
    help="Mistral API key. If not provided, will use MISTRAL_API_KEY environment variable.",
    envvar="MISTRAL_API_KEY",
)
@click.option(
    "--model", 
    help="Mistral OCR model to use.", 
    default="mistral-ocr-latest"
)
@click.option(
    "--json/--no-json",
    "-j/-J",
    "json_output",
    is_flag=True,
    default=False,
    help="Return raw JSON instead of markdown text/Return markdown text (default).",
)
@click.option(
    "--html/--no-html",
    "-h/-H",
    is_flag=True,
    default=False,
    help="Convert markdown to HTML/Keep as markdown (default).",
)
@click.option(
    "--inline-images/--no-inline-images",
    "-i/-I",
    is_flag=True,
    default=False,
    help="Include images inline as data URIs/Don't include inline images (default).",
)
@click.option(
    "--extract-images/--no-extract-images",
    "-e/-E",
    is_flag=True,
    default=True,  # Extract images by default
    help="Extract images as separate files (default)/Skip extracting images.",
)
@click.option(
    "--silent/--verbose",
    "-s/-v",
    is_flag=True,
    default=False,
    help="Suppress all output except for the requested data/Show detailed progress (default).",
)
@click.option(
    "--pages",
    type=int,
    default=20,
    help="Limit processing to the first N pages (default: 20).",
)
def arxiv_to_markdown(
    arxiv_url,
    file_path,
    output_file,
    api_key,
    model,
    json_output,
    html,
    inline_images,
    extract_images,
    silent,
    pages,
):
    """Process a PDF file and convert it to markdown using Mistral OCR.
    
    Either ARXIV_URL (URL of the arXiv paper) or --file-path (path to local PDF) must be provided.
    The script will process the PDF with OCR and save the result in the papers/ directory.
    
    \b
    Examples:
      python arxiv_ocr.py https://arxiv.org/abs/1706.03762 --api-key YOUR_API_KEY
      python arxiv_ocr.py --file-path path/to/paper.pdf --pages 5 --html
    """
    # Validate inputs
    if not arxiv_url and not file_path:
        raise click.ClickException("Either ARXIV_URL or --file-path must be provided.")
    if arxiv_url and file_path:
        raise click.ClickException("Cannot provide both ARXIV_URL and --file-path.")
    
    # Validate API key
    if not api_key:
        raise click.ClickException("No API key provided and MISTRAL_API_KEY environment variable not set.")
    
    try:
        # Check if papers directory exists, create if not
        papers_dir = Path("papers_ocr_results")
        papers_dir.mkdir(exist_ok=True)
        
        # Handle local file or arXiv URL
        if file_path:
            # Use the filename (without extension) as the paper title
            paper_title = file_path.stem
            submission_date = None
            arxiv_id = None
            bibtex_content = None
            
            # Read the PDF file
            if not silent:
                click.echo(f"Reading local PDF file: {file_path}", err=True)
            pdf_content = file_path.read_bytes()
        else:
            # Extract arXiv ID from URL
            if not silent:
                click.echo(f"Extracting arXiv ID from URL: {arxiv_url}", err=True)
            arxiv_id = extract_arxiv_id(arxiv_url)
            if not silent:
                click.echo(f"Found arXiv ID: {arxiv_id}", err=True)
            
            # Get paper metadata
            if not silent:
                click.echo("Fetching paper metadata...", err=True)
            paper_title, submission_date = get_paper_metadata(arxiv_id)
            
            # Download PDF
            if not silent:
                click.echo(f"Downloading PDF from arXiv...", err=True)
            pdf_content = get_arxiv_pdf(arxiv_id)
            
            # Download BibTeX citation
            if not silent:
                click.echo(f"Downloading BibTeX citation...", err=True)
            bibtex_content = get_arxiv_bibtex(arxiv_id)
            if bibtex_content:
                if not silent:
                    click.echo(f"BibTeX citation retrieved successfully", err=True)
            else:
                if not silent:
                    click.echo(f"Could not retrieve BibTeX citation", err=True)
        
        sanitized_title = sanitize_filename(paper_title)
        if not silent:
            click.echo(f"Paper title: {paper_title}", err=True)
            if submission_date:
                click.echo(f"Submission date: {submission_date}", err=True)
            click.echo(f"Sanitized filename: {sanitized_title}", err=True)
        
        # Create temp file for PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf_path = Path(temp_pdf.name)
            temp_pdf.write(pdf_content)
        
        try:
            if not silent:
                click.echo(f"Created temporary PDF at {temp_pdf_path}", err=True)
            
            # Process PDF with Mistral OCR
            client = Mistral(api_key=api_key)
            uploaded_file = None
            
            try:
                # Upload PDF to Mistral
                if not silent:
                    click.echo(f"Uploading file to Mistral...", err=True)
                uploaded_file = client.files.upload(
                    file={
                        "file_name": f"{paper_title}.pdf",
                        "content": pdf_content,
                    },
                    purpose="ocr",
                )
                
                # Get signed URL
                signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
                
                # Process with OCR
                if not silent:
                    click.echo(f"Processing with OCR model: {model}...", err=True)
                
                # Prepare OCR processing parameters
                ocr_params = {
                    "document": DocumentURLChunk(document_url=signed_url.url),
                    "model": model,
                    "include_image_base64": True,  # Always request images
                }
                
                # Add pages parameter if limited pages are requested
                if pages > 0:
                    ocr_params["pages"] = list(range(pages))
                    if not silent:
                        click.echo(f"Limiting processing to first {pages} pages", err=True)
                
                pdf_response = client.ocr.process(**ocr_params)
                
                # Parse response
                response_dict = json.loads(pdf_response.model_dump_json())
                
                # Define output paths - always create a directory structure
                output_dir = papers_dir / sanitized_title
                output_dir.mkdir(exist_ok=True)
                
                # Determine output filename
                if output_file:
                    # If output_file is provided, use it directly
                    output_filename = output_file
                else:
                    # Default to sanitized_title with appropriate extension
                    output_filename = f"{sanitized_title}.{'html' if html else 'md'}"
                
                output_file = output_dir / output_filename
                bibtex_file = output_dir / f"{sanitized_title}.bib"
                
                # For HTML output, ensure it has .html extension
                if html and not output_file.suffix.lower() == '.html':
                    output_file = output_file.with_suffix('.html')
                
                # Save BibTeX citation if available
                if bibtex_content:
                    try:
                        bibtex_file.write_text(bibtex_content)
                        if not silent:
                            click.echo(f"BibTeX citation saved to {bibtex_file}", err=True)
                    except Exception as e:
                        if not silent:
                            click.echo(f"Error saving BibTeX file: {str(e)}", err=True)
                
                # Process images if needed
                image_map = {}
                if extract_images or inline_images:
                    image_count = 0
                    
                    # For extract_images, we need a directory
                    if extract_images:
                        image_dir = output_dir
                    
                    # Look for images in the OCR response
                    for page in response_dict.get("pages", []):
                        for img in page.get("images", []):
                            if "id" in img and "image_base64" in img:
                                image_id = img["id"]
                                image_data = img["image_base64"]
                                
                                # Sometimes the base64 data has a data URI prefix, sometimes not
                                if image_data.startswith("data:image/"):
                                    # Extract the mime type and base64 data
                                    mime_type = image_data.split(";")[0].split(":")[1]
                                    base64_data = image_data.split(",", 1)[1]
                                else:
                                    # Determine mime type from file extension or default to jpeg
                                    ext = image_id.split(".")[-1].lower() if "." in image_id else "jpeg"
                                    mime_type = f"image/{ext}"
                                    base64_data = image_data
                                
                                # For extracted images, save to disk
                                if extract_images:
                                    # Create a suitable filename if it doesn't have an extension
                                    if "." not in image_id:
                                        ext = mime_type.split("/")[1]
                                        image_filename = f"{image_id}.{ext}"
                                    else:
                                        image_filename = image_id
                                    
                                    image_path = image_dir / image_filename
                                    
                                    try:
                                        with open(image_path, "wb") as img_file:
                                            img_file.write(base64.b64decode(base64_data))
                                        
                                        # Map image_id to relative path for referencing
                                        image_map[image_id] = image_filename
                                        image_count += 1
                                    except Exception as e:
                                        if not silent:
                                            click.echo(f"Warning: Failed to save image {image_id}: {str(e)}", err=True)
                                
                                # For inline images, prepare data URIs
                                elif inline_images:
                                    # Ensure it has the data URI prefix
                                    if not image_data.startswith("data:"):
                                        image_data = f"data:{mime_type};base64,{base64_data}"
                                    
                                    image_map[image_id] = image_data
                    
                    if not silent and extract_images and image_count > 0:
                        click.echo(f"Extracted {image_count} images to {image_dir}", err=True)
                
                # Generate output content
                if json_output:
                    result = json.dumps(response_dict, indent=4)
                else:
                    # Concatenate markdown content from all pages
                    markdown_contents = [
                        page.get("markdown", "") for page in response_dict.get("pages", [])
                    ]
                    markdown_text = "\n\n".join(markdown_contents)
                    
                    # Add metadata at the top
                    markdown_text = f"# {paper_title}\n\n"
                    
                    # Add source information near the top
                    markdown_text += f"*Source: [arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})*\n\n"
                    
                    # Add submission date if available
                    if submission_date:
                        markdown_text += f"*[Submitted on {submission_date}]*\n\n"
                    
                    # Add the content
                    content_text = "\n\n".join(markdown_contents)
                    markdown_text += content_text
                    
                    # Add link to BibTeX if available at the bottom
                    if bibtex_content:
                        markdown_text += f"\n\n---\n*[BibTeX citation]({sanitized_title}.bib)*\n"
                    
                    # Post-processing: Remove duplicate title if present
                    lines = markdown_text.split('\n')
                    if len(lines) >= 3:
                        if lines[0].strip().startswith('#') and lines[2].strip().startswith('#') and lines[0].strip() == lines[2].strip():
                            # Remove duplicate title
                            lines.pop(2)
                            markdown_text = '\n'.join(lines)
                    
                    # Handle image references
                    for img_id, img_src in image_map.items():
                        pattern = r"!\[(.*?)\]\(\s*" + re.escape(img_id) + r"\s*\)"
                        replacement = r"![\1](" + img_src + r")"
                        markdown_text = re.sub(pattern, replacement, markdown_text)
                    
                    if html:
                        # Convert markdown to HTML
                        md = markdown.Markdown(extensions=["tables"])
                        html_content = md.convert(markdown_text)
                        
                        # Add HTML wrapper with basic styling
                        result = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{paper_title}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 20px;
        }}
        img {{ max-width: 100%; height: auto; }}
        h1, h2, h3 {{ margin-top: 1.5em; }}
        p {{ margin: 1em 0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
                    else:  # markdown
                        result = markdown_text
                
                # Write output to file
                output_file.write_text(result)
                
                if not silent:
                    click.echo(f"Results saved to {output_file}", err=True)
                    if bibtex_content:
                        click.echo(f"BibTeX citation saved to {bibtex_file}", err=True)
                    click.echo(f"Original arXiv URL: {arxiv_url}", err=True)
                    click.echo(f"PDF URL: https://arxiv.org/pdf/{arxiv_id}.pdf", err=True)
            
            finally:
                # Clean up uploaded file
                if uploaded_file:
                    try:
                        client.files.delete(file_id=uploaded_file.id)
                        if not silent:
                            click.echo("Temporary Mistral file deleted", err=True)
                    except Exception as e:
                        if not silent:
                            click.echo(f"Warning: Could not delete temporary Mistral file: {str(e)}", err=True)
        
        finally:
            # Clean up temp file
            if temp_pdf_path.exists():
                os.unlink(temp_pdf_path)
                if not silent:
                    click.echo("Temporary PDF file deleted", err=True)
    
    except Exception as e:
        raise click.ClickException(f"Error: {str(e)}")


if __name__ == "__main__":
    arxiv_to_markdown()