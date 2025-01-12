from pydantic import BaseModel, ValidationError
from typing import Literal, Optional
from ollama import chat


# Define the data model
class disability_certificate(BaseModel):
    document_type: Literal[
        "disability_certificate", "other_document"  # Specify valid document types
    ]
    name_of_disbled: str
    #name_of_disability: str
    


def disability_certificate_extract_event_information(formatted_text: str) -> Optional[dict]:
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
Aim: To analyze the provided document and predict if the text is from a disability_certificate or not.

Procedure:
- The goal is to classify whether the text is a disability certificate or another type of document.
- If the text is an achievement certificate, extract the following details:
  - Name of Disabled
- If the document is not a disability certificate, return the document type as 'other_document' and leave other fields empty.

The data to analyze: 
""" + formatted_text

        # Send the chat request
        response = chat(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3.3:70b",
            format=disability_certificate.model_json_schema(),
            options={'temperature': 0.2},
        )

        # Parse and validate the response
        output_json = disability_certificate.model_validate_json(response.message.content)
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
    BILL OF SUPPLY Scan QR code for kiosk payment COMMERCIAL BILL DATE METER STATUS CYCLE NUMBER TYPE OF SUPPLY SHREE LAXMI DEVELOPERS 30-Dec-2024 Active 11 SINGLE PHASE G-19 ZOOM PLAZA, METER ROOM NO. 2, L T ROAD, BORIVALI WEST OPP GORAI BUS DEPOT, MUMBAI 400092 TARIFF CONNECTION DATE SANCTIONED LOAD (KW) BILL NUMBER LT II (A) 12-09-2015 101266376934 90**********80 1.00 Mobile sanj******er@gmail.com Emai BILL DISTRIBUTION NO. BILLING STATUS PREVIOUS READING DATE PRESENT READING DATE PAN Boriwali/Shimpoli/ 27-Dec-2024 27-Nov-2024 Regular GST 11/212/01B/001/001 Bill Month Units Consumed Current Month Bill Previous Outstanding CA NO. 151663501 the premises for which the power supply has been granted is an authorised structure
ll amount to proof of ownership of the premises." ₹5.72 ₹1107.35 Dec-24 45  ₹1110.00 Bill Period: 28-Nov-2024 - 27-Dec-2024 Previous Units : 37 Due Date: 20-Jan-2025 ● Round sum payable by discount date 06-Jan-2025: Amt ₹1100.00 Discount ₹9.03 The due date refers to only current bill amount, • Round sum payable after  due date 20-Jan-2025 : Amt ₹1120.00 DPC ₹13.84 previous balance is payable immediately Nearest Collection Centre (Cash/Cheque) Manoj Chouhan Scan code to pay your bill via (use any UPI app) Division Head - Borivali Adani Electricity, Receiving Station, S.V.Road,Shimpoli, Borivali (West) Mumbai-400092 MAJOR BILL COMPONENTS (Rounded off amt) CONSUMPTION TREND Previous year Current year NET OTHER(Cr) 2 48 5 NET PREV  રેણ FAC 51 117 24 WHEELING  204 DUTIES/TAXES 12 258 ENERGY 36 46 12 5 O FIXED 475 Nov Sep May Dec Oct Aug Jul Jun Apr Mar Feb Jan s bill for power supply cannot be treated or utilised as pr
would the issuance                would the issuance O 119 238 357 476 HELP CENTER METER DETAILS Present Previous Multiplying Meter Consumption Units(kWh) 19122 Toll Free No.(24X7) www.adanielectricity.com Reading Reading Factor Number helpdesk.mumbaielectricity@adani.com 8823256 6534.00 6579.00 45 1 Adani Electricity ,Swami Vivekananda road, Kandivali west , 9 Mumbai-400067 Whatsapp Us on : 9594519122 For power interruption complaint or restoration status 1. Missed Call on 9594519122 from your Registered Mobile No 2. SMS POWER <9 digit account no.> to 9594519122" from your Registered Mobile No For internal complaint redressal system(ICRS), visit our website: www.adanielectricity.com Total Consumption 45 Join us on: IMPORTANT MESSAGE ● As per Honorable MERC approval dated 30th October 2024, Fuel adjustment charge(FAC) is being levied in current month. For any query, kindly connect at our Toll free number:19122 or visit https://www.adanielectricity.com/faqs for details. ● Please note that all important communication related to your account are being sent on 90*****80 registered with us. In case of any change, do inform us immediately to avoid any inconvenience and enjoy our uninterrupted services ● Tentative meter reading date for your JAN-25 bill is 27/01/2025 To ensure you never miss any 71-266 electricity related alerts and notifications, Register / update your phone number and Email ID right away. SCAN HERE E. 80.E. CONSOLDATED STAMP DUTY PAID BY ORDER NO. LO#ENF-2/CSD/76/2024 Validity Period Dt. 13/09/2024 to Dt. 30/09/2022 / OW 4516 DT. 13 /09/2024, GRN NO MH005567162024254E, DT D5769/A5769/B71/S70/R5769 18/07/2024,SBI / DEFACE NO 003217443202425, DEFACE DT 29/07/2024
    """
    result = extract_event_information(sample_text)
    if result:
        print("Extracted Information:", result)
    else:
        print("Failed to extract information.")

    result = extract_event_information(sample_text)
    if result:
        print("Extracted Information:", result)
    else:
        print("Failed to extract information.")