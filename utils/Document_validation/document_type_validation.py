from pydantic import BaseModel
from typing import Literal, Optional
from ollama import chat


# Define the data model
class ValidDocumentType(BaseModel):
    Document_type: Literal[
        "Mark Sheet", "Others"
    ]  # Specify valid document types
    class_: Optional[Literal[
        "Higher Secondary Certificate Examination", 
        "Secondary School Certificate/Central Board of Secondary Education/Intermediate", 
        "Bachelors Degree", 
        "Master's Degree",
        "None"
    ]]
    

def extract_event_information(text: str) -> dict:
    """
    Extracts event information from the input text using OpenAI's API.

    Args:
        text (str): The input text containing document information.

    Returns:
        dict: Parsed document type and additional information.
    """
    response = chat(
        messages = [
            {
                'role': 'user',
                'content': "Extract all the important data from the data provided. \n\n Data: {text}"

            }
        ],
        model='llama3.1:8b',
        format = ValidDocumentType.model_json_schema()
    )

    outut_json = ValidDocumentType.model_validate_json(response.message.content)
    print(outut_json)

    return outut_json

# Example usage
if __name__ == "__main__":
    sample_text = """--- Page 1 ---
0266.
BeEE
JBoard of Jntermediate Ebucation
6,M~7R
ANDHRA PRADESH, INDIA
J3ov
7
CS.6o
~"o eocnis. .
SI.No P147112
 ,7nECiFJon;.
06/06022/P147112
F;6?Co.
Fp
6C
FL .
BCAEF
Vec
E
. | .=
CEn
J '^
XOn?oof
Fxe
=7k0
.
e
F :
INTERMEDIATE
Aadhaar
No:
PASS CERTIFICATE CUM MEMORANDUM OF MARKS
369511858726
This is t0 certify that KALAPATI PUSHPATEJ
Father Name
KALAPATI SUBRAHMANYAM
Mother Name
KALAPATI NAGA RANI
bearing
Registered No: 1606232368
has appeared at the Intermediate Public
Examination held in MARCH-2016
and passed in A GRADE
with ENGLISH
as the medium of instruction.
The subjects in which
HE
was examined and the marks awarded are as follows
YYear
Year
Subject
Maximum
Marks
Maximum
Marks
Marks
Secured
Marks
Secured
Part
4
ENGLISH
100
083
100
081
Part
2
SANSKRIT
100
098
100
087
Part
3
Optional Subjects
MATHEMATICS - A
075
058
075
056
MATHEMATICS
B
075
056
075
060
PHYSICS
060
054
060
048
CHEMISTRY
060
049
060
057
PHYSICS PRACTICAL
030
030
CHEMISTRY PRACTICAL
030
030
ENVIRONMENTAL EDUCATION
0
U A
F
IE
D
ETHICS AND HUMAN VALUES
Q
U A
F
Ie
D
Total Marks
In Figures
LQua
In words
'EIGHT "FOUR * SEVEN*
Date
19-04-2016
6 n[
R
Fo
Naad
t
PRINCIPAL
3I 
Halanua}
0
VcNSco-OPEKATIVE
83818
0
84cb &
8 
Iature of b1e'
Seal
Controller of Examinatlons
VA n Signaturovot Saeiorci
Roba rze
RuLE3RR2 OVERLEAF "INDICATES MARKS OBTAINED AT AN EARLIER EXAMINATION.
http:llapbie cgg gov inlgetOrData.do?rno=1606232368
2L6042
n

--- Page 1 ---
Board of Intermediate Education ANDHRA PRADESH, INDIA SI.No. P147112 Aadhaar No 36951 1858725 T CALL COLLECT OF AND INDU This is to certify that KALAPATI PUSHPATEJ Father Name : KALAPATI SUBRAHMANYAM Mother Name : KALAPATI NAGA RANI bearing has appeared at the Intermediate Public Registered No. 1606232368 Examination held in MARCH-2016 and passed in A GRADE as the medium of instruction. with ENGLISH was examined and the marks awarded are as follows : The subjects in which HE rear Subject Maximum Maximum Marks Marks Marks Secured Part - 1 : ENGLISH 100 081 083 100 Part - 2 : 087 SANSKRIT 100 098 100 Part - 3 : Optional Subjects MATHEMATICS - A 075 058 075 056 MATHEMATICS - B 075 075 060 056 PHYSICS 060 054 060 048 CHEMISTRY 049 060 060 057 PHYSICS PRACTICAL 030 030 CHEMISTRY PRACTICAL 030 030 ENVIRONMENTAL EDUCATION A D O D E ETHICS AND HUMAN VALUES D Q A B Total Marks In Figures In words *EIGHT**FOUR ** SEVEN* 19-04-2016 Date PRINCIPAL VIGNAN'S CO-OPERATIONS Controller of Examinations Signature of the Principal and College Sea OVERLEAF *INDICATES MARKS OBTAINED AT AN EARLIER EXAMINATION. 546BL 2 http://apbie.cgg.gov.in/getQrData.do?rno=1606232368

"""

    result = extract_event_information(sample_text)
    print(result)
