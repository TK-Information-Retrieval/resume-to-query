from summarize import LLMModel
from pdf2text import extract_v1

QUERY_TEMPLATE = """
Please summarize the following resume into a concise search query that captures the candidate's key skills, experience, and job roles.

Resume:
{resume}
"""

def pipeline(path):
    resume_text = None
    try:
        resume_text = extract_v1(path)
    except Exception as e:
        print(f"Error extracting resume: {e}")
        return None

    llm = LLMModel()
    prompt = QUERY_TEMPLATE.format(resume=resume_text)
    
    try:
        response = llm.generate_response(prompt)
        return response
    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return None

if __name__ == "__main__":
    result = pipeline("./resume.pdf")
    if result:
        print(result)
