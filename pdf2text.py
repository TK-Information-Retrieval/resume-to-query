from pdfminer.high_level import extract_text
from PIL import Image
import fitz  # PyMuPDF
import easyocr
import numpy as np

RESUME_PATH = "resume.pdf"

# resume text based
def extract_v1(path):
  res = extract_text(path)
  return res

# resume w/ ocr
def extract_v2(path):
  reader = easyocr.Reader(['en'])
  doc = fitz.open(path)
  full_text = ""
  for page in doc:
      pix = page.get_pixmap(dpi=300)
      img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
      result = reader.readtext(np.array(img), detail=0)
      full_text += "\n".join(result) + "\n"
  return full_text

if __name__ == '__main__':
  print("====================== pdfminer ======================")
  print(extract_v1(RESUME_PATH))
  print(f"\n======================================================\n")
  print("====================== pdf2image + tesseract ======================")
  print(extract_v2(RESUME_PATH))
