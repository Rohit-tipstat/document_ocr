from ollama import chat
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

class Mark_for_each_subject(BaseModel):
    semeste_year_class: str | None = Field(
        default=None,
        description="The semester year or class to which the subject belongs(Eg., 'Semester-3', 'Year-1', 'Class 7')."
    )
    subject_name: str = Field(
        ...,
        description="The name of the subject(Eg., 'Mathematics', 'Science', 'Statistics')"
    )
    subject_marks_grade_obtained: str = Field(
        ...,
        description="The marks or grade obtained by in the subject(Eg., '84', 'B+', 'C')"
    )
    subject_maximum_marks: int | None = Field(
        default=None,
        description="The maximum marks for that subject if applicable"
    )

class Marksheet(BaseModel):
    student_name: str = Field(..., description="Name of the student.")
    school_college_name: str = Field(..., description="Name of the school or college.")
    register_name: str = Field(..., description="Registration number of the student.")
    course_id: str | None = Field(None, description="The unique ID of the course.")
    course_name: str | None = Field(None, description="The name of the course.")
    class_semester_year: str = Field(..., description="Class, semester, or year information of the student's marksheet.")
    year_of_passing: str | None = Field(None, description="Year in which the course was completed.")
    maximum_total_marks: int = Field(
        ..., description="The total maximum marks for all subjects."
    )
    obtained_total_marks: int = Field(
        ..., description="The total marks obtained by the student."
    )
    grading_type: Literal['percentage', 'CGPA'] = Field(
        ..., description="The type of grading used: percentage or CGPA."
    )
    percentage: str | None = Field(
        None, description="The percentage score of the student, if applicable."
    )
    grade: str | None = Field(None, description="The overall grade of the student.")
    remark: str | None = Field(None, description="Any additional remarks.")
    cgpa: str | None = Field(None, description="The CGPA score, if applicable.")
    education_board: Literal['CBSE', "ICSE", "StateBoard"] = Field(
        ..., description="The education board affiliated with the marksheet."
    )
    subject: List[Mark_for_each_subject] = Field(
        ..., description="List of marks for each subject."
    )

response = chat(
    messages=[
        {
            'role': 'user',
            'content': """You are a helpful assistant, help me extract the important information from marksheet data provided below. 
            The data is extracted from marksheet using OCR tools and the data might not be in a proper format. 
            Make sure there are no simple mistakes like obtained marks being greater than the maximum marks.
            Do not makeup any data.
            
            
            Here is the extracted data:
            --- Page 1 ---
22005500
22/22093/J372753
171
SL. No. J372753
Board of Intermediate Education,A R
Bhavan, Nampally; Hyderabad
500 001
WE
INTERMEDIATE
PASS CERTIFICATE CUM MEMORANDUM OF MARKS
This is to certify that
MOTHILAL NAIK N
son
of
SUKHYA NAIK N
bearing
Registered No
1022222324
has appeared at the Intermediate
Public
Examination held in
MARCH-201O
and passed in
GRADE
with
ENGLISH
as the
Medium of Instruction_
The subjects in which
he
was examined and the marks awarded are as follows
LYear
Ycar
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
ENGLISH
100
085
100
088
Part
2
SANSKRIT
100
090
100
098
Part
Optional Subjects
MATHEMATICS - A
075
073
075
073
MATHEMATICS
B
075
070
075
070
PHYSICS
060
049
060
049
CHEMISTRY
060
053
060
049
PHYSICS PRACTICAL
030
030
CHEMISTRY PRACTICAL
030
027
ENVIRONMENTAL EDUCATION
e  D
Total Marks
In Figures
904
In words
*NINE  ZERO"**FOUR"
Date
30-04-2010
X@
781
Reevo_sd
~l_Aw
Aiartyte efthe Principal and Cdlege5nr
Controller of Examinations
NNOTE
ELIGIBiI!Y RULES EAVEEN
INDICATES MARKS OBTAINED AT AN EARLIER EXAMINATION
EB
ba[
1022222324
Vidya
cet
"""
        }
    ],
    model = 'llama3.3:80b',
    format = Marksheet.model_json_schema(),
)

result = Marksheet.model_validate_json(response.message.content)