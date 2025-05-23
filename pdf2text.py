from pdfminer.high_level import extract_text

RESUME_PATH = "resume.pdf"

# resume text based
def extract_v1(path):
  res = extract_text(path)
  return res

if __name__ == '__main__':
  print(extract_v1(RESUME_PATH))
