from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional, Literal, Any
from langgraph.graph import StateGraph
from langgraph.graph import END

from app.services.logger import setup_logger
from app.api.error_utilities import LoaderError, ToolExecutorError
from app.utils.document_loaders import get_docs
from app.services.schemas import ChatMessage

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv, find_dotenv

from app.tools.rubric_generator.core import executor as rubric_generator_executor
from app.tools.rubric_generator.tools import RubricCriteria
from app.tools.writing_feedback_generator.core import executor as writing_feedback_generator_executor
from app.tools.writing_feedback_generator.tools import WritingFeedback # Type of writing_feedback_generator's result

from app.utils.actions_for_assistants.actions_for_assistants import (
    generate_questions_to_json, 
    go_to_process_questions_json, 
    process_content
)
load_dotenv(find_dotenv())

logger = setup_logger()

chat_google_genai = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)

    with open(absolute_file_path, 'r') as file:
        return file.read()

#================================CLASSES DEFINITION================================
#----------------------ESSAY GRADING----------------------
class CriterionGrading(BaseModel):
    criterion: RubricCriteria = Field(..., description="The rubric criterion being evaluated, including its descriptions and each description's corresponding points.")
    grade: int  = Field(..., description="The grade assigned for this criterion. It should be how many points out of the point scale")
    reasoning: str = Field(..., description="Explanation of why this grade was assigned to this criterion.")

class GradingOutput(BaseModel):
    criteria_grading: List[CriterionGrading] = Field(..., description="The list of assigned grade and explaination for said grade for each rubric criterion.")
    total_grade: str

#----------------------AGGREGRATED ESSAY GRADING and FEEDBACK----------------------
class EssayGradingAndFeedbackResult(BaseModel):
    criterion: RubricCriteria
    grading: int
    feedback: WritingFeedback
class EssayGradingResult(BaseModel):
    grading_and_feedback_criteria: List[EssayGradingAndFeedbackResult]
    total_grade: str

#----------------------GRAPH STATE----------------------
class GraphState(TypedDict):
    user_query: str
    action: str
    chat_history: list[ChatMessage]
    assistant_system_message: str

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
    criteria: str   # To be filled with processed Rubric Generator's Output
    writing_to_review: str
    #criteria_file_url: str
    #criteria_file_type: str
    wtr_file_url: str   # Essay to Review file url
    wtr_file_type: str  # Essay to Review file type

    #--------Intermediate Outputs--------
    rubric_criteria: List[RubricCriteria]    # Generate Rubric Output
    grading_output: GradingOutput    # Essay Grading Output
    #generated_feedbacks: List[WritingFeedback]   # List of Criterias + its corresponding feedback

    #--------Final Outputs--------
    essay_grading_result: EssayGradingResult  # Final Output

    result: Any

#================================ESSAY GRADING ASSISTANT PIPELINE================================
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
    rubric_objectives:
        direct text -> str
        url -> str
    essay:
        direct text -> str
        url -> str
    lang
Generate Rubric:
    grade_level = grade_level
    point_scale = point_scale
    objectives = rubric_objectives
    objectives_file_url
    objectives_file_type
    assignment_desc = assignment_desc
    lang = lang
Generate Feedback:
    grade_level = grade_level
    assignmetn_description = assignment_desc
    criteria = objectives
    writing_to_review = essay
    wtr_file_url
    wtr_file_type
    lang = lang

generate_rubric: 
    criterias + aux -> generate_rubric => RubricOutput
generate_grading:
    grade_level
    point_scale
    assignment_desc
    RubricOutput + Essay -> genereate_grading => EssayGradingOutput
generate_feedback:
    essay_grading_result.total_grade = EssayGradingOutput.total_grade
    
    for grading in EssayGradingOutput.criteria_grading
        grading + essay -> generate_feedback => WritingFeedback
        AggregratedGradingResult:
            criterion: grading.criterion
            grading: grading.grading
            feedback: WritingFeedback

        essay_grading_result.append(AggregratedGradingResult)
'''
#------------------------Essay Grading Node------------------------
"""
Input:
    grade_level
    point_scale
    assignment_desc
    rubric_criteria: RubricOutput
    writing_to_review:  Need context if empty
    wtr_file_url:
    wtr_file_type:
"""
class EssayGradingGeneratorArgs(BaseModel):
    grade_level: Literal["pre-k", "kindergarten", "elementary", "middle", "high", "university", "professional"]
    point_scale: int
    assignment_desc: str
    rubric_criteria: List[RubricCriteria]
    writing_to_review: str
    wtr_file_url: str
    wtr_file_type: str

class EssayGradingGeneratorPipeline:
    def __init__(self, args=None, verbose=False):
        self.verbose = verbose
        self.args = args
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.vectorstore_class = Chroma
        self.parsers = {
            "criterion_grading": JsonOutputParser(pydantic_object=CriterionGrading)
        }
        self.vectorstore = None
        self.retriever = None

    def compile_vectorstore(self, documents: List[Document]):
        if self.verbose:
            logger.info("Creating vectorstore from documents...")
        self.vectorstore = self.vectorstore_class.from_documents(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        self.retriever = self.vectorstore.as_retriever()
        if self.verbose:
            logger.info("Vectorstore and retriever created successfully.")

    def compile_pipeline(self):
        base_prompt = PromptTemplate(
            template=(
                "Analyze the provided writing and evaluate it based on the rubric criterion '{criterion}'. "
                "Assign grading and reasoning behind the grading using the descriptions for each point section: {criterion_description}. "
                "Use the provided Writing to Review: {writing_to_review}. If it is empty, use the context {context}."
                "Assignment Description: {assignment_description}. Grade Level: {grade_level}. Point Scale: {point_scale}"
                "Respond in this JSON format: \n{format_instructions}"
            ),
            input_variables=["assignment_description", "grade_level", "point_scale", "criterion", "criterion_description", "writing_to_review", "context"],
            partial_variables={"format_instructions": self.parsers["criterion_grading"].get_format_instructions()}
        )
    
        chains = {}

        for criterion in self.args.rubric_criteria:
            criterion_description_str = "\n".join([
                f"{desc.points} points: " + " ".join(desc.description)
                for desc in criterion.criteria_description
            ])

            prompt = PromptTemplate(
                template=base_prompt.template,
                input_variables=base_prompt.input_variables,
                partial_variables={
                    **base_prompt.partial_variables,
                    "criterion": criterion.criteria,
                    "criterion_description": criterion_description_str
                }
            )

            chain = prompt | self.model | self.parsers["criterion_grading"]

            chains[criterion.criteria] = chain
        
        return RunnableParallel(branches=chains)

    def generate_context(self, query: str) -> str:
        return self.retriever.invoke(query)

    def generate_grading(self, documents: Optional[List[Document]] = None):
        if documents:
            self.compile_vectorstore(documents)
            context = self.generate_context("Provide context for grading this assignment")
        else:
            context = ""
        
        pipeline = self.compile_pipeline()
        inputs = {
            "grade_level": self.args.grade_level,
            "point_scale": self.args.point_scale,
            "assignment_description": self.args.assignment_desc,
            "rubric_criteria": self.args.rubric_criteria,
            "writing_to_review":  self.args.writing_to_review,
            "context": context
        }

        try:
            results = pipeline.invoke(inputs)
            grading_output = GradingOutput(
                criteria_grading=[
                    results["branches"][criterion.criteria]
                    for criterion in self.args.rubric_criteria
                ],
                # Total grade: total of assigned grade / total of maximum grade for each criteria (i.e point scale * # of criteria)
                total_grade = f"{sum(results["branches"][criterion.criteria]['grade'] for criterion in self.args.rubric_criteria)} / {self.args.point_scale * len(self.args.rubric_criteria)}"
            )

            if self.verbose:
                logger.info("(Grade Essay Node) Grading successfully generated.")
            return grading_output

        except Exception as e:
            logger.error(f"(Grade Essay Node) Error in generating grading: {e}")
            raise ValueError("(Grade Essay Node) Failed to generate grade.")

def grade_essay(state: GraphState): # Act as Executor for Grade Essay Node
    try:
        if (state["wtr_file_type"]):
            logger.info(f"Generating Writing To Review docs. from {state["wtr_file_url"]}")
        
        docs = None

        def fetch_docs(file_url, file_type):
            return get_docs(file_url, file_type, True) if file_url and file_type else None
        
        docs = fetch_docs(state["wtr_file_url"], state["wtr_file_type"])

        essay_grading_generator_args = EssayGradingGeneratorArgs(
            grade_level=state["grade_level"],
            point_scale=state["point_scale"],
            assignment_desc=state["assignment_desc"],
            rubric_criteria=state["rubric_criteria"],
            writing_to_review=state["writing_to_review"],
            wtr_file_url=state["wtr_file_url"],
            wtr_file_type=state["wtr_file_type"]
        )

        grading_output = EssayGradingGeneratorPipeline(args=essay_grading_generator_args).generate_grading(docs)

        logger.info(f"(Essay Grading Assistant) (Grade Essay Node) Essay Grading generated successfully.")

    except LoaderError as e:
        error_message = e
        logger.error(f"(Essay Grading Assistant) (Grade Essay Node) Error in Essay Grade Genarator Pipeline: {error_message}")
        raise ToolExecutorError(error_message)
    
    except Exception as e:
        error_message = f"(Essay Grading Assistant) (Grade Essay Node) Error during running Essay Grading Generator: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    return {
        "grading_output": grading_output
    }

#------------------------Generate Rubric Node------------------------
def generate_rubric(state: GraphState):
    try:
        rubric_output = rubric_generator_executor(grade_level=state["grade_level"],
                                        point_scale=state["point_scale"],
                                        objectives=state["objectives"],
                                        assignment_desc=state["assignment_desc"],
                                        objectives_file_url=state["objectives_file_url"],
                                        objectives_file_type=state["objectives_file_type"],
                                        ad_file_url="",
                                        ad_file_type="",
                                        lang=state["lang"])
    except LoaderError as e:
        error_message = e
        logger.error(f"(Essay Grading Assistant) (Generate Rubric Node) Error in Rubric Genarator Pipeline: {error_message}")
        raise ToolExecutorError(error_message)

    except Exception as e:
        error_message = f"(Essay Grading Assistant) (Generate Rubric Node) Error during running Rubric Generator: {e}"
        logger.error(error_message)
        raise ValueError(error_message)

    return {
        "rubric_criteria": rubric_output["criterias"]
    }

#------------------------Generate Feedback Node------------------------
def generate_feedback(state: GraphState):
    try:
        if (state["essay_grading_result"]):
            essay_grading_result = state["essay_grading_result"]
        else:
            essay_grading_result = EssayGradingResult(
                grading_and_feedback_criteria=[],
                total_grade=state["grading_output"].total_grade
            )

        for assigned_grading in state["grading_output"].criteria_grading:
            criterion_grading = f"""
                The criterion being considered is {assigned_grading.criterion.criteria}
                The rubric description for this criterion:
            """
            for desc in assigned_grading.criterion.criteria_description:
                criterion_grading += f"""
                    Points: {desc.points}
                    Description: {desc.description}
                """
            criterion_grading += f"""
                    The grading for this criterion is {assigned_grading.grade}
                    The reasoning behind why the essay receives this grading for this criterion is: {assigned_grading.reasoning}
                    Construct your feedback with consideration for the criterion being considered, its description, its corresponding grading and reasoning behind said grading.
                """

            generated_feedback = writing_feedback_generator_executor(grade_level=state["grade_level"],
                                                assignment_description=state["assignment_desc"],
                                                criteria=criterion_grading,
                                                writing_to_review=state["writing_to_review"],
                                                criteria_file_url="",
                                                criteria_file_type="",
                                                wtr_file_url=state["wtr_file_url"],
                                                wtr_file_type=state["wtr_file_type"],
                                                lang=state["lang"])

            essay_and_feedback_result = EssayGradingAndFeedbackResult(
                criterion=assigned_grading.criterion,
                grading=assigned_grading.grade,
                feedback=generated_feedback
            )

            # Append to the final result
            essay_grading_result.grading_and_feedback_criteria.append(essay_and_feedback_result)

    except LoaderError as e:
        error_message = e
        logger.error(f"(Essay Grading Assistant) (Generate Feedback Node) Error in Writing Feedback Generator Pipeline: {error_message}")
        raise ToolExecutorError(error_message)
    
    except Exception as e:
        error_message = f"(Essay Grading Assistant) (Generate Feedback Node) Error during running Writing Feedback Generator: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    return {
        "essay_grading_result": essay_grading_result
    }

#================================LANGGRAPH WORKFLOW================================
workflow = StateGraph(GraphState)

# Dummy entry node to provide the start point to build conditional edges to process_content or rubric_generator
def workflow_entry(state):
    pass
workflow.add_node("entry", workflow_entry)
workflow.set_entry_point("entry")

def go_to_process_content_or_generate_rubric(state):
    """
    Directs the workflow to 'generate_rubric' if the action is 'essay_grading'.
    Otherwise, it continues to 'process_content'.
    """
    if state['action'] == 'essay_grading':
        return 'generate_rubric'  # Skip 'process_content' and go directly to 'generate_rubric'
    else:
        return 'process_content'

workflow.add_conditional_edges(
    "entry",
    go_to_process_content_or_generate_rubric,
    {
        'process_content': 'process_content',
        'generate_rubric': 'generate_rubric',
    }
)

# 5 Universal actions: Translate, summarize, rewrite, question generation and custom prompts
# start ---if 'action' != 'essay_grading'---> process_content
workflow.add_node("process_content", process_content)
workflow.add_node("generate_questions_to_json", generate_questions_to_json)
workflow.add_conditional_edges(
    "process_content",
    go_to_process_questions_json,
    {
        'generate_questions_to_json': 'generate_questions_to_json',
        END: END
    }
)

# Essay Grading Assistant workflow:
# start ---if 'action' == 'essay_grading'---> generate_rubric ---> grade_essay ---> generate_feedback
workflow.add_node("generate_rubric", generate_rubric)
workflow.add_node("grade_essay", grade_essay)
workflow.add_node("generate_feedback", generate_feedback)

workflow.add_edge("generate_rubric", "grade_essay")
workflow.add_edge("grade_essay", "generate_feedback")
workflow.add_edge("generate_feedback", END)

def compile_essay_grading_assistant():
    app = workflow.compile()
    return app