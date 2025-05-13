from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(api_key= os.environ['GROQ_API_KEY'], 
               model = 'llama-3.1-8b-instant',
               temperature = 0.7)

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

def parse_with_groq(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm 
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start = 1):
        response = chain.invoke({'dom_content': chunk, 'parse_description': parse_description})

        print(f"Parsed batch {i} of {len(dom_chunks)}")
        print(response)
        parsed_results.append(response.content)

    return "\n".join(parsed_results)

