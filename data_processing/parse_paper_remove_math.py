from lxml import etree
import os
import re

def extract_text_from_xml(xml_file, exclude_sections=None):
    """
    Extract text from XML file, organized by headers, with option to exclude sections.
    Also cleans up mathematical symbols and notation.
    
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
        title_text = clean_text(title_text)
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
            header_text = clean_text(header_text)
            
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
                p_text = clean_text(p_text)
                
                # Skip paragraphs that are mostly mathematical notation
                if is_mostly_math(p_text):
                    continue
                    
                if p_text:
                    text_content.append(p_text)
    
    return '\n\n'.join(text_content)

def clean_text(text):
    """Clean up text by removing/replacing mathematical symbols and notation."""
    # Remove angle brackets with content inside (like ⟨w τ , x τ,i ⟩)
    text = re.sub(r'⟨[^⟩]*⟩', '', text)
    
    # Remove other mathematical symbols
    text = re.sub(r'[∈∀∃∧∨¬∩∪⊂⊃⊆⊇≤≥≠±×÷→←↔∞∂∫∑∏√∇]', '', text)
    
    # Remove subscript and superscript notation
    text = re.sub(r'[τ,][,i][,q]', '', text)
    
    # Replace Greek letters with their names
    greek_letters = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
        'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
        'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
        'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
    }
    
    for greek, name in greek_letters.items():
        text = text.replace(greek, name)
    
    # Remove expressions like "N (0, I d×d)"
    text = re.sub(r'[A-Z]\s*\([^)]*\)', '', text)
    
    # Remove matrix notation like "R d×d"
    text = re.sub(r'R\s*\d+×\d+', '', text)
    
    # Remove or replace other mathematical notation as needed
    text = re.sub(r'\s+', ' ', text)  # Clean up whitespace
    
    return text.strip()

def is_mostly_math(text):
    """Check if a text is mostly mathematical notation."""
    # Count mathematical symbols and characters
    math_symbols = set('⟨⟩∈∀∃∧∨¬∩∪⊂⊃⊆⊇≤≥≠±×÷→←↔∞∂∫∑∏√∇τ')
    math_count = sum(1 for char in text if char in math_symbols)
    
    # Count formulas like "N (0, I)" or "R d×d"
    formula_count = len(re.findall(r'[A-Z]\s*\([^)]*\)', text))
    formula_count += len(re.findall(r'R\s*\d+×\d+', text))
    
    # If more than 20% of the text is mathematical, consider it mostly math
    total_length = len(text.strip())
    if total_length == 0:
        return True
        
    math_ratio = (math_count + formula_count * 5) / total_length  # Weigh formulas more heavily
    
    return math_ratio > 0.2  # Adjust threshold as needed

def main():
    # Process all XML files in the pdfs directory
    pdfs_dir = 'arxiv_papers'
    
    # Sections to exclude
    exclude_sections = ["Related Work", "References", "Bibliography", "Acknowledgements", 
                        "Acknowledgement", "Impact Statement", "Appendix"]
    
    for filename in os.listdir(pdfs_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(pdfs_dir, filename)
            print(f"Processing {filename}...")
            text = extract_text_from_xml(xml_path, exclude_sections)
            output_path = os.path.join(pdfs_dir, filename.replace('.xml', 'math_removed.txt'))
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Extracted text saved to {output_path}")

if __name__ == "__main__":
    main()