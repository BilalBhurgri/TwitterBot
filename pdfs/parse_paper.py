import xml.etree.ElementTree as ET
import os
import re

def get_clean_text(element):
    """Extract text from element while excluding specific tags"""
    text_parts = []
    
    # If this is a ref or formula tag, skip it
    if element.tag.endswith('ref') or element.tag.endswith('formula'):
        return ''
        
    # Add the element's direct text
    if element.text:
        text_parts.append(element.text)
    
    # Process child elements
    for child in element:
        # Skip ref and formula tags
        if child.tag.endswith('ref') or child.tag.endswith('formula'):
            continue
        # Recursively get text from other children
        text_parts.append(get_clean_text(child))
        # Add the tail text (text after this element)
        if child.tail:
            text_parts.append(child.tail)
    
    return ''.join(text_parts)

def parse_xml_paper(xml_path):
    """Parse XML paper and extract text from title and paragraph tags, excluding references"""
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Print the root tag to see what we're dealing with
        print(f"Root tag: {root.tag}")
        
        # Get all namespaces
        namespaces = {k: v for k, v in root.attrib.items() if k.startswith('xmlns')}
        print(f"Found namespaces: {namespaces}")
        
        # Initialize text content
        text_content = []
        
        # Extract title - try different possible paths
        title = None
        for path in ['.//title', './/{*}title', './/article-title', './/{*}article-title', './/front/title', './/{*}front/{*}title']:
            title = root.find(path)
            if title is not None:
                print(f"Found title using path: {path}")
                break
        
        if title is not None and title.text:
            print(f"Title text: {title.text.strip()}")
            text_content.append(title.text.strip())
        else:
            print("No title found")
        
        # Extract paragraphs - try different possible paths
        paragraphs = []
        for path in ['.//p', './/{*}p', './/sec/p', './/{*}sec/{*}p', './/body//p', './/{*}body//{*}p']:
            found = root.findall(path)
            if found:
                print(f"Found {len(found)} paragraphs using path: {path}")
                paragraphs.extend(found)
        
        print(f"Total paragraphs found: {len(paragraphs)}")
        
        # Track if we're in a reference section
        in_ref_section = False
        
        for p in paragraphs:
            # Check if we're entering or leaving a reference section
            parent = p.getparent() if hasattr(p, 'getparent') else p.find('..')
            if parent is not None:
                parent_tag = parent.tag.lower()
                if 'ref' in parent_tag or 'references' in parent_tag:
                    in_ref_section = True
                    print("Found reference section")
                    continue
                elif in_ref_section and 'sec' in parent_tag:
                    in_ref_section = False
                    print("Left reference section")
            
            # Skip if we're in a reference section
            if in_ref_section:
                continue
                
            # Get clean text content, excluding ref and formula tags
            p_text = get_clean_text(p).strip()
            
            if not p_text:
                continue
                
            print(f"Processing paragraph: {p_text[:100]}...")
            
            # Filter out words with special characters but keep basic punctuation
            words = p_text.split()
            filtered_words = []
            for word in words:
                # Keep words that are alphanumeric or contain only basic punctuation
                if all(c.isalnum() or c in ".,;'\"!?-:" for c in word):
                    filtered_words.append(word)
            
            if filtered_words:
                text_content.append(' '.join(filtered_words))
        
        # Join all text with newlines
        final_text = '\n'.join(text_content)
        
        print(f"Total text content length: {len(final_text)}")
        
        # Save the extracted text
        output_path = xml_path.replace('.xml', '.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        print(f"Saved extracted text to {output_path}")
        
        return final_text
        
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        import traceback
        traceback.print_exc()
        return ""

def main():
    # Process all XML files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('.xml'):
            print(f"\nProcessing {filename}...")
            parse_xml_paper(filename)

if __name__ == "__main__":
    main() 