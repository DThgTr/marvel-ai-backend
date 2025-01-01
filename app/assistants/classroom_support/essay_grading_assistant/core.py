from app.assistants.classroom_support.essay_grading_assistant.tools import (
    compile_essay_grading_assistant,
    read_text_file
)
from app.services.logger import setup_logger
from app.services.schemas import ChatMessage, Message

logger = setup_logger()

'''
Input:
    grade_level
    point_scale
    assigment_desc
    rubric_objectives
    rubric_objectives_file_url
    rubric_objectives_file_type
    essay
    essay_file_url
    essay_file_type
    lang

Preprocessing:
    grade_level
    point_scale
    assignment_desc
    objectives:
        direct text -> str
        url -> str
    essay:
        direct text -> str
        url -> str
    lang
'''

def executor(
        grade_level: str,
        point_scale: str,
        assignment_desc: str,
        rubric_objectives: str,
        rubric_objectives_file_url: str,
        rubric_objectives_file_type: str,
        writing_to_review: str,
        writing_file_url: str,
        writing_file_type: str,
        lang: str,
        action: str,
        messages: list[Message]=None,
        k=3
    ):

    logger.info(f'Generating response from Essay Grading Assistant - Action: [{action}]')

    print(messages)
    
    chat_context = [
        ChatMessage(
            role=message["role"], 
            type=message["type"], 
            text=message["payload"]["text"]
        ) for message in messages[-k:]
    ]

    essay_grading_assistant = compile_essay_grading_assistant()
    user_query = messages[-1]["payload"]["text"]

    """
    #--------External Tools--------
    # Common Inputs:
    grade_level: str
    assignment_desc: str
    lang: str

    # Rubric Generator Inputs
    point_scale: int
    objectives: str
    objectives_file_url: str
    objectives_file_type: str
    #ad_file_url: str
    #ad_file_type: str

    # Writing Feedback Inputs
    criteria: str
    writing_to_review: str
    #criteria_file_url: str
    #criteria_file_type: str
    wtr_file_url: str   # Essay to Review file url
    wtr_file_type: str  # Essay to Review file type

    #--------Intermediate Results--------
    rubric_output: RubricOutput
    essay_grading_output: EssayGradingOutput
    #generated_feedbacks: List[WritingFeedback]   # List of Criterias + its corresponding feedback

    essay_grading_result: AggregratedGradingResult
    """
    inputs = {
        "user_query": user_query,
        "action": action,
        "chat_history": chat_context,
        "assistant_system_message": read_text_file('prompt/essay_grading_assistant_context.txt'),

        # Common Inputs:
        "grade_level": grade_level,
        "assignment_desc": assignment_desc,
        'lang': lang,

        # Rubric Generator Inputs
        "point_scale": point_scale,
        "objectives": rubric_objectives,
        "objectives_file_url": rubric_objectives_file_url,
        "objectives_file_type": rubric_objectives_file_type,

        # Writing Feedback Inputs
        "criteria": "",
        "writing_to_review": writing_to_review,
        "wtr_file_url": writing_file_url,   # Essay to Review file url
        "wtr_file_type": writing_file_type,  # Essay to Review file type

         #--------Intermediate Outputs--------
        "rubric_output": None,
        "essay_grading_output": None,

        #--------Final Outputs--------
        "essay_grading_result": None
    }
    
    result = essay_grading_assistant.invoke(inputs)

    logger.info(f"Response generated successfully for CoTeacher - Action: [{action}]")

    if (action == "essay_grading"):
        return result["essay_grading_result"]
    return result["result"]