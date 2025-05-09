from summarize import LLMModel
from pdf2text import extract_v1, extract_v2
from utils import clean_resume_text, clean_ocr_resume_text

QUERY_TEMPLATE = """
Analyze the following resume and extract key details in the Skills.

Then, based on the extracted information, generate a concise and effective search engine query that summarizes the candidate`s professional profile and expertise in ONE SENTENCE ONLY.

Resume:
{resume}
"""

def pipeline(path):
    resume_text = None
    try:
        resume_text = extract_v2(path)
    except Exception as e:
        print(f"Error extracting resume: {e}")
        return None

    llm = LLMModel()
    prompt = QUERY_TEMPLATE.format(resume=resume_text)
    
    try:
        initial_response = llm.generate_response(prompt)
        clean_response = initial_response.split("</think>")[-1].strip()

        final_prompt = f"summarize into one sentence: {clean_response}"
        final_response = llm.generate_response(final_prompt)
        return final_response.split("</think>")[-1].strip()
    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return None

if __name__ == "__main__":
    result = pipeline("./resume.pdf")
    if result:
        print(result)
