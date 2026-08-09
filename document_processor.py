import os
import fitz  # PyMuPDF
from docx import Document


# ------------------------------------------
# Extract text from PDF
# ------------------------------------------
def extract_pdf(file_path):
    try:
        text = ""
        
        pdf = fitz.open(file_path)
        
        for page in pdf:
            page_text = page.get_text().strip()
            
            # Skip blank pages
            if page_text:
                text += page_text + "\n"
        
        pdf.close()
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading PDF file: {e}")


# ------------------------------------------
# Extract text from DOCX
# ------------------------------------------
def extract_docx(file_path):
    try:
        doc = Document(file_path)
        
        text = ""
        
        for paragraph in doc.paragraphs:
            paragraph_text = paragraph.text.strip()
            
            # Skip empty paragraphs
            if paragraph_text:
                text += paragraph_text + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading DOCX file: {e}")


# ------------------------------------------
# Extract text from TXT
# ------------------------------------------
def extract_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        if not text:
            raise Exception("TXT file is empty.")
        
        return text
    
    except FileNotFoundError:
        raise Exception(f"TXT file not found: {file_path}")
    
    except UnicodeDecodeError:
        raise Exception(f"TXT file encoding error. Please ensure the file is UTF-8 encoded.")
    
    except Exception as e:
        raise Exception(f"Error reading TXT file: {e}")


# ------------------------------------------
# Main Extraction Function
# ------------------------------------------
def extract_text(file_path):
    # Check if file exists first
    if not os.path.exists(file_path):
        raise Exception(f"File not found: {file_path}")
    
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension == ".pdf":
        return extract_pdf(file_path)
    
    elif extension == ".docx":
        return extract_docx(file_path)
    
    elif extension == ".txt":
        return extract_txt(file_path)
    
    else:
        raise Exception(f"Unsupported file type: {extension}. Supported types: .pdf, .docx, .txt")


# ------------------------------------------
# Testing
# ------------------------------------------
if __name__ == "__main__":
    
    file_path = input("Enter file path: ")
    
    try:
        text = extract_text(file_path)
        
        print("\n✅ Extracted Text\n")
        print("=" * 60)
        print(text)
        print("=" * 60)
        print(f"\n📊 Statistics:")
        print(f"   Characters: {len(text)}")
        print(f"   Words: {len(text.split())}")
        print(f"   Lines: {len(text.splitlines())}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")