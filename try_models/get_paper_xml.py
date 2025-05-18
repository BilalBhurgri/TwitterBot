# USE GROBID!
import requests
import os
import argparse

grobid_url = "http://localhost:8080"

def process_pdf(pdf_path):
    """Extract structured data from a PDF using GROBID"""
    # Endpoint for full text processing
    process_url = f"{grobid_url}/api/processFulltextDocument"
    
    # Check if file exists
    if not os.path.isfile(pdf_path):
        return {"error": f"File not found: {pdf_path}"}
    
    # Prepare the file for upload
    with open(pdf_path, 'rb') as pdf_file:
        files = {'input': (os.path.basename(pdf_path), pdf_file, 'application/pdf')}
        
        # Optional parameters (you can customize these)
        params = {
            'consolidateHeader': '1',
            'consolidateCitations': '1',
            'teiCoordinates': '0',  # Set to 1 if you need coordinates
            'includeRawAffiliations': '0'
        }
        
        # Make the request to GROBID
        response = requests.post(process_url, files=files, params=params)
    
    if response.status_code == 200:
        # Success - return the TEI XML content
        return response.text
    else:
        # Error handling
        return {
            "error": f"GROBID request failed with status code {response.status_code}",
            "details": response.text
        }

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from a PDF using GROBID')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file')
    args = parser.parse_args()
    print("Using GROBID URL:", grobid_url)

    status_url = f"{grobid_url}/api/isalive"
    response = requests.get(status_url)
    if response.status_code == 200:
        print("GROBID is running")
    else:
        print("GROBID is not running")
    result = process_pdf(args.pdf_path)
    # Either save to file or process further
    with open(args.pdf_path.replace(".pdf", "") + ".xml", "w", encoding="utf-8") as out_file:
        out_file.write(result)