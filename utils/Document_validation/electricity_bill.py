from pydantic import BaseModel, ValidationError
from typing import Literal, Optional
from ollama import chat


# Define the data model
class electricity_bill(BaseModel):
    document_type: Literal[
        "electricity_bill", "other_documents"  # Specify valid document types
    ]
    name_of_the_owner: str


def electricity_bill_extract_event_information(formatted_text:str) -> Optional[dict]:
    """
    Extracts event information from the input text using OpenAI's API.
    Args:
        text (str): The input text containing document information.
    Returns:
        dict: Parsed document type and additional information or None if an error occurs.
    """
    try:
        # Prepare the prompt
        prompt = """
Aim: To analyze the provided document and predict if the text is from an electricity bill or not.

Procedure:
- The goal is to classify whether the text is an electricity bill or another type of document.
- If the text is an electricity bill, extract the following details:
  - Consumer name
- If the document is not an electricity bill, return the document type as 'other_document' and leave other fields empty.

The data to analyze: 
""" + formatted_text
        # Send the chat request
        response = chat(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3.1:8b",
            format=electricity_bill.model_json_schema(),
            options={'temperature': 0.1},
        )

        # Parse and validate the response
        output_json = electricity_bill.model_validate_json(response.message.content)
        #output_json = response.message.content
        
        # print("Validation successful:", output_json)
        return output_json.dict()
        #return output_json


    except ValidationError as ve:
        print("Validation error:", ve)
    except Exception as e:
        print("An error occurred during processing:", e)

    return None



# Example usage
if __name__ == "__main__":
    sample_text = """
    West Bengal Council of Higher Secondary Education Mark Sheet**

* **Institution Code:** B334818
* **Roll Number:** 315411
* **Examination Year:** 2024
* **Student Name:** Swarnendu Darbar

**Subject-wise Marks and Grades:**

1. **Compulsory Language:**
        * **Bengali (BNGA):** 
                + Theory: 76
                + Practical: 20
                + Total: 96 (Ninety Six)
                + Grade: O
        * **English (ENGB):** 
                + Theory: 76
                + Practical: 20
                + Total: 96 (Ninety Six)
                + Grade: O
2. **Compulsory Elective:**
        * **Chemistry (CHEM):** 
                + Theory: 64
                + Practical: 30
                + Total: 94 (Ninety Four)
                + Grade: O
        * **Mathematics (MATH):** 
                + Theory: 78
                + Practical: 20
                + Total: 98 (Ninety Eight)
                + Grade: O
        * **Physics (PHYS):** 
                + Theory: 53
                + Practical: 30
                + Total: 83 (Eighty Three)
                + Grade: A+
3. **Optional Elective:**
        * **Biology (BIOS):** 
                + Theory: 60
                + Practical: 30
                + Total: 90 (Ninety)
                + Grade: O

**Grand Total:** 474
**Result:** Passed
**Overall Grade:** O

**Authorized Signature:**
* **Deputy Secretary (Examination):** Utpal Kr. Biswas
    """
    result = extract_event_information(sample_text)
    if result:
        print("Extracted Information:", result)
    else:
        print("Failed to extract information.")
