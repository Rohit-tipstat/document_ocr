from ollama import chat
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from utils.ocr_tools.surya_ocr_tool import get_image_text_suryaocr
from utils.ocr_tools.easy_ocr_tool import get_image_text_easyocr


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
        ..., description="The total maximum marks for all subjects, if applicable"
    )
    obtained_total_marks: int = Field(
        ..., description="The total marks obtained by the student. This might sometimes be written as Total marks"
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


image_path = "/root/document_ocr/images/12th/Document_2_App_1.pdf"

# extracting info using Two OCR-tools
# 1) Using EasyOCR
easy_ocr_text_extracted_ = get_image_text_easyocr(image_path)
print("Easy OCR text extractor: \n", easy_ocr_text_extracted_)


# 2) Using SuryaOCR
surya_ocr_text_extracted = get_image_text_suryaocr(image_path)
print("Surya OCR text extractor: \n", surya_ocr_text_extracted)

response = chat(
    messages=[
        {
            'role': 'user',
            'content': """You are a helpful assistant, help me extract the important information from marksheet data provided below. 
            The data is extracted from marksheet using OCR tools and the data might not be in a proper format. 
            Make sure there are no simple mistakes like obtained marks being greater than the maximum marks.
            Do not makeup any data and dont assume anything, Extract and return the information that is in the data.
            
            Output format:
            1) Structure the data in a tabular format.
            2) Make no assumptions about the data.
            3) Return whatever information is available in the data provided.
            4) We are using two OCR extractor to extract text for better understanding. 


            Here is the extracted data from two OCRs:
            1) Surya OCR: {surya_ocr_text_extracted}

            2) Easy OCR: {easy_ocr_text_extracted_}
"""
        }
    ],
    model = 'llama3.3',
    format = Marksheet.model_json_schema(),
)

result = Marksheet.model_validate_json(response.message.content)
print(result)
