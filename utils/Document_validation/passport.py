from pydantic import BaseModel, ValidationError
from typing import Literal, Optional
from ollama import chat


# Define the data model
class passport(BaseModel):
    document_type: Literal[
        "passport_document", "other_document"  # Specify valid document types
    ]
    name: str
    
def passport_extract_event_information(formatted_text: str) -> Optional[dict]:
    """
    Extracts event information from the input text using Llama.
    Args:
        text (str): The input text containing document information.
    Returns:
        dict: Parsed document type and additional information or None if an error occurs.
    """
    try:
        # Prepare the prompt
        prompt = """
Aim: To analyze the provided document and predict if the text is from an passport or not.

Procedure:
- The goal is to classify whether the text is a passport or another type of document.
- If the text is a passport, extract the following details:
  - Name
- If the document is not a passport, return the document type as 'other_document' and leave other fields empty.

The data to analyze: 
""" + formatted_text

        # Send the chat request
        response = chat(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3.3:70b",
            format=passport.model_json_schema(),
            options={'temperature': 0.2},
        )

        # Parse and validate the response
        output_json = passport.model_validate_json(response.message.content)
        print("Validation successful:", output_json)
        return output_json.dict()

    except ValidationError as ve:
        print("Validation error:", ve)
    except Exception as e:
        print("An error occurred during processing:", e)

    return None



# Example usage
if __name__ == "__main__":
    sample_text = """
    केन्द्रीय माध्यमिक शिक्षा वोर्ड, दिल्ली पजीकरण स Central Board of Secondary Education, Delhi Registration No. : M115/06026/0092 क्रम राख्या - 288102 माध्यमिक विद्यालय परीक्षा (सत्र : 2013-15 S.No.SSE/2015 SECONDARY SCHOOL EXAMINATION (SESSION : 2013 - 15) ग्रेड शीट सह निष्पादन प्रमाण पत्र Grade Sheet cum Certificate of Performance यह प्रमाणित किया जाता है कि This is to certify that MATTA NAGASAI DEEPAK अनुक्रमांक Roll No. : 4068241 माता/पिता/संरक्षक का नाम MATTA PHANI KUMARI / MATTA SURYANARAYANA Mother's/Father's/Guardian's Name जन्म तिथि Date of Birth 13/01/2000 13TH JANUARY TWO THOUSAND विद्यालय School 06026-DEFENCE LAB SCHOOL KANCHANBAGH HYDERABAD TL का निष्पादन निम्नानुसार रहा has performed as follows भाग Part-1 शैक्षिक क्षेत्र Scholastic Areas 1. शैक्षणिक निष्पादन Academic Performance : क्रेस Class D 68 Class विषय कोड तथा नाम Grade Gradi Grade Subject Code and Name FA 5A FA SA 104 ENGLISH COMM. C1 C1 C1 06 B4 BI B1 08 007 TELUGU A2 B1 B1 08 B1 B1 A2 08 041 MATHEMATICS C2 E1 C2 ** 05 A2 C2 B2 ** 07 086 SCIENCE B4 C1 B2 07 B1 C2 B2 07 087 SOCIAL SCIENCE C2 B1 C1 B2 B1 B2 ** 07 07 ice of Grade Point and Per Additional : 7.4 संधित ग्रेड बिन्दू का औसत (सीजीपीए) Cumulative Grade Point Average (CGPA) : * कथन और श्रवण कौशलो (एएसएल) के आकलन में ग्रेड Grade in Assessment of Speaking and Listening Skills (ASL) : CLASS IX - A2 CLASS X - A2 भाग Part - 2 : सह-शैक्षिक कार्य क्षेत्र Co-Scholastic Are 2 (a) (A) जीवन कोशल Life Skills : कक्षा Class IX जीवन कोशल ग्रेड qial Class X गोड Life Skills Grade वर्णनात्मक उल्लेख Descriptive Indicator Grade वर्णनात्मक उल्लेख Descriptive Indicators Identifies personal strengths and weaknesses and uses them to arrive at Identifies personal strengths and weaknesses, analyses a problem with चितन कोशल meaningful decisions relevant information and usually chooses appropriate alternatives and B B Thinking Skills makes meaningful decisions. Empathetic, Displays sensitivity towards differently-abled students, possesses Interpersonal and communicative skills are satisfactory and usually takes सामाजिक कौशल good interpersonal skills and appreciates other's opinions, accepts feedback feedback and criticism positively. A Social Skills C from teachers, elders and peers for self-improvement. Self-confident, optimistic, manages personal challenges and adverse situations Identifies weaknesses, stress and negative emotions fairly well, manages भावनात्मक कोशल effectively and constructively, handles stress well, expresses emotions them with self confidence and is empathetic. A Emotional Skills B appropriately and readily takes help when needed. 2 (ख)(B) कार्य शिक्षा Work Education : Grasps assigned tasks easily, self-motivated, helpful, guides others and is Innovative, with excellent grasp of any assignment and is very punctual in B punctual.  the completion of set task, self-motivated, empathetic, inspires others and A कार्य शिक्षा an excellent team worker. Readily shoulders responsibility. Work Education 2 (ग)(C) दृश्य और प्रदर्शन कला Visual and Performing Arts Participates actively in artistic activities, creative, very observant, appreciates Open to learning, participates in artistic activities and demonstrates दृश्य और प्रदर्शन कलाएं B and understands various art forms. originality in some art forms. D Visual and Performing Arts : 2 (ध)(D) अभिवृत्तियां एवं मूल्य Attitudes and Values के प्रति towards ग्रेड वर्णनात्मक उल्लेख Descriptive Indicators ग्रे ख वर्णनात्मक उल्लेख Descriptive Indicator Grade Grade Very courteous to teachers and elders, adheres to school rules, sincere and Very courteous towards teachers, follows school rules, has a positive helpful towards teachers, has a positive attitude to learning, com attitude and takes criticism in the right spirit. B Teachers easily with and confides in teachers, accepts feedback and criticism positively. Expresses ideas and opinions with clarity, is sensitive and supportive towards Expresses ideas and interacts effectively in class, sensitive towards peers सरपादी peers and differently-abled schoolmates, receptive to new ideas and and differently abled schoolmates, respects new ideas and opinions, gets A Schoolmates B suggestions, inspires others and manages diversity well. along well with peers. An enthusiastic participant in various school programmes and environmental  Participates in various school programmes and environmental initiatives विद्यालय कार्यक्रम initiatives, possesses leadership skills. Usually takes in pride in the school and और प्रयोगरण regularly, possesses good leadership qualities and is punctual. B B School Programmes & respects school property. Environment Understands value systems, abides by rules and regulations. Ethical and always Understands value systems quite well and adheres to school rules, respects courteous towards peers and elders, respects the national flag and symbols, मुल्य प्रणालिया the national flag and symbols. Honest, courteous and sensitive to diversity, A B sensitive to diversity and shows empathy towards the disadvantaged. Value Systems with a positive outlook. भाग Part - 3 सह पाठ्यक्रम कार्यकलाप Co-Curricular Activities. 3 (क)(A) सह पाठ्यक्रम कार्यकलाप Co-Curricular Activities : कायकलाप Activity Applies science to everyday life, participates in scientific activities at inter- and Scientific Skills Participates in scientific activities and events at the school level, observant intra mural events, displays good laboratory skills and is very observant. with good laboratory skills. B C Information and Actively participates in computer technology related events at the school and Actively takes initiative to organize & participate in computer technology Communication inter school levels, handles IT equipment with ease, shows keen interest and is Technology(ICT) Skills related activities at the inter- & intra-mural events,very observant & a good B A very observant. decision maker,has an innovative & practical approach. 3 (ख)(B) स्वास्थ्य एवं शारीरिक शिक्षा Health and Physical Education : कायकलाप Activity Good in an identified sport and represents the school at various levels, has Sports /Indigenous Talented in an identified sport, represents the school at various levels, has Sports (Kho-Kho Etc.) excellent hand-eye co-ordination, exhibits agility, endurance and flexibility. stamina, strength and flexibility with good hand- eye coordination, displays A demonstrates sporting skills, team spirit and determination to excel. team spirit, discipline and punctuality. A Is aware of types of the plants and the time of the year during which they are Interested and understands the techniques, postures (mudras) and is good X- Gardening / Shramdaan grown. Shows keen interest in gardening and can look after plants well. Enjoys at breath regulation exercises, flexible and agile and can meditate. B A and exhibits a desire to learn. Readily takes part in shramdaan. Integrates the discipline with practical, day to day activities. < - Yoga उन्नत ग्रेड Upgraded Grade परिणाम Result: QUALIFIED FOR ADMISSION TO HIGHER CLASSES ## Upscaled Grade leeel Delhi दिनांक Dated 28-05-2015 Controller of Examinations
    """
    result = passport_extract_event_information(sample_text)
    if result:
        print("Extracted Information:", result)
    else:
        print("Failed to extract information.")
