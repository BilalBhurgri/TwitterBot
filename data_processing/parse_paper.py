from lxml import etree
import os
import re

def extract_text_from_xml(xml_file, exclude_sections=None):
    """
    Extract text from XML file, organized by headers, with option to exclude sections.
    
    Args:
        xml_file: Path to XML file
        exclude_sections: List of section names to exclude (case-insensitive)
    """
    if exclude_sections is None:
        exclude_sections = ["Related Work", "References", "Bibliography", "Acknowledgements"]
    
    # Convert exclude_sections to lowercase for case-insensitive comparison
    exclude_sections = [section.lower() for section in exclude_sections]
    
    # Parse the XML file
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(xml_file, parser)
    root = tree.getroot()
    
    # Define namespace
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    
    text_content = []
    
    # Extract the main title from the header
    titles = root.xpath('//tei:titleStmt/tei:title', namespaces=ns)
    if titles:
        title_text = titles[0].text.strip()
        text_content.append(f"# {title_text}")
    
    # Find the body tag
    body = root.xpath('//tei:body', namespaces=ns)
    if not body:
        return "No body found in the document"
    
    # Find all div elements (usually represent sections)
    divs = body[0].xpath('./tei:div', namespaces=ns)
    
    for div in divs:
        # Get the header for this div
        headers = div.xpath('./tei:head', namespaces=ns)
        
        if headers:
            header = headers[0]
            header_str = etree.tostring(header, encoding='unicode')
            header_str = re.sub(r'<ref[^>]*?>.*?</ref>', '', header_str, flags=re.DOTALL)
            header_str = re.sub(r'<formula[^>]*?>.*?</formula>', '', header_str, flags=re.DOTALL)
            header_text = re.sub(r'<[^>]+>', '', header_str).strip()
            
            # Check if this section should be excluded
            if any(exclude.lower() in header_text.lower() for exclude in exclude_sections):
                print(f"Excluding section: {header_text}")
                continue
            
            # Add the header
            n_attr = header.get('n', '')
            if n_attr:
                section_header = f"## {n_attr} {header_text}"
            else:
                section_header = f"## {header_text}"
                
            text_content.append(section_header)
            
            # Get all paragraphs within this div
            paragraphs = div.xpath('./tei:p', namespaces=ns)
            
            for p in paragraphs:
                p_str = etree.tostring(p, encoding='unicode')
                p_str = re.sub(r'<ref[^>]*?>.*?</ref>', '', p_str, flags=re.DOTALL)
                p_str = re.sub(r'<formula[^>]*?>.*?</formula>', '', p_str, flags=re.DOTALL)
                p_text = re.sub(r'<[^>]+>', '', p_str).strip()
                
                if p_text:
                    text_content.append(p_text)
    
    return '\n\n'.join(text_content)

def main():
    # Process all XML files in the pdfs directory
    pdfs_dir = 'pdfs'
    
    # Sections to exclude
    exclude_sections = ["Related Work", "References", "Bibliography", "Acknowledgements", 
                        "Acknowledgement", "Impact Statement", "Appendix"]
    
    for filename in os.listdir(pdfs_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(pdfs_dir, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_xml(xml_path, exclude_sections)
            output_path = os.path.join(pdfs_dir, filename.replace('.xml', '.txt'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Extracted text saved to {output_path}")

if __name__ == "__main__":
    main()