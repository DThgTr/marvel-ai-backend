import pytest

from app.assistants.classroom_support.essay_grading_assistant.core import executor

"""
core.executor parameters:
    grade_level: str,
    point_scale: int,
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
"""
empty_args = {
    "grade_level": None,
    "point_scale": None,
    "assignment_desc": None,
    "rubric_objectives": None,
    "rubric_objectives_file_url": None,
    "rubric_objectives_file_type": None,
    "writing_to_review": None,
    "writing_file_url": None,
    "writing_file_type": None,
    "lang": None,
}

grading_base_args = {
    "grade_level": "university",
    "point_scale": 4,
    "assignment_desc": "Write an essay on Linear Regression",
    "lang": "en",
    "messages": [
            {
                "role":"human",
                "type":"text",
                "timestamp":"string",
                "payload":{
                    "text":""
                }
            }
        ]
}
"""
Args for testing:
    "rubric_objectives": None,
    "rubric_objectives_file_url": None,
    "rubric_objectives_file_type": None,
    "writing_to_review": None,
    "writing_file_url": None,
    "writing_file_type": None,
"""

#=========================================ESSAY GRADING ACTION TESTS=========================================
def test_executor_grading_wtr_url_pdf_valid():
    result = executor(
        **grading_base_args,
        action="essay_grading",
        rubric_objectives="Understand the concepts of vector spaces, linear transformations, and matrix operations.",
        rubric_objectives_file_url="",
        rubric_objectives_file_type="",
        writing_to_review="",
        writing_file_url="https://firebasestorage.googleapis.com/v0/b/kai-ai-f63c8.appspot.com/o/uploads%2F510f946e-823f-42d7-b95d-d16925293946-Linear%20Regression%20Stat%20Yale.pdf?alt=media&token=caea86aa-c06b-4cde-9fd0-42962eb72ddd",
        writing_file_type="pdf"
    )
    print(f"test_executor_grading_wtr_url_pdf_valid => Result:\n{result}")
    assert isinstance(result, str)

#=========================================AI ASSISTANT UNIVERSAL ACTION TESTS=========================================
def test_executor_translate_valid():
    result = executor(
        **empty_args,
        action="translate",
        messages=[
            {
                "role":"human",
                "type":"text",
                "timestamp":"string",
                "payload":{
                    "text":"""Please, translate 'Large Language Models (LLMs) are advanced artificial intelligence systems trained on vast amounts 
                    of text data to understand and generate human-like language. These models leverage deep learning techniques, particularly transformer 
                    architectures, to process and generate text across a wide range of contexts and tasks. LLMs are capable of performing diverse functions, 
                    including language translation, summarization, content creation, and even complex problem-solving. Their capabilities continue to expand as 
                    they are fine-tune for specific applications, making them invaluable tools in industries such as healthcare, education, customer support, 
                    and software development.' from English to Spanish."""
                }
            }
        ]
    )
    assert isinstance(result, str)

def test_executor_translate_invalid():
    with pytest.raises(ValueError) as exc_info:
        executor(
            **empty_args,
            action="translate123",
            messages=[
                {
                    "role":"human",
                    "type":"text",
                    "timestamp":"string",
                    "payload":{
                        "text":"""Please, translate 'Large Language Models (LLMs) are advanced artificial intelligence systems trained on vast amounts 
                        of text data to understand and generate human-like language. These models leverage deep learning techniques, particularly transformer 
                        architectures, to process and generate text across a wide range of contexts and tasks. LLMs are capable of performing diverse functions, 
                        including language translation, summarization, content creation, and even complex problem-solving. Their capabilities continue to expand as 
                        they are fine-tune for specific applications, making them invaluable tools in industries such as healthcare, education, customer support, 
                        and software development.' from English to Spanish."""
                    }
                }
            ]
        )
    assert isinstance(exc_info.value, ValueError)

def test_executor_summarize_valid():
    result = executor(
        **empty_args,
        action="summarize",
        messages=[
            {
                "role":"human",
                "type":"text",
                "timestamp":"string",
                "payload":{
                    "text":"""Please, summarize 'Large Language Models (LLMs) are advanced artificial intelligence systems trained on vast amounts 
                    of text data to understand and generate human-like language. These models leverage deep learning techniques, particularly transformer 
                    architectures, to process and generate text across a wide range of contexts and tasks. LLMs are capable of performing diverse functions, 
                    including language translation, summarization, content creation, and even complex problem-solving. Their capabilities continue to expand 
                    as they are fine-tune for specific applications, making them invaluable tools in industries such as healthcare, education, customer 
                    support, and software development."""
                }
            }
        ]
    )
    assert isinstance(result, str)

def test_executor_summarize_invalid():
    with pytest.raises(ValueError) as exc_info:
        executor(
            **empty_args,
            action="summarize123",
            messages=[
                {
                    "role":"human",
                    "type":"text",
                    "timestamp":"string",
                    "payload":{
                        "text":"""Please, summarize 'Large Language Models (LLMs) are advanced artificial intelligence systems trained on vast amounts 
                        of text data to understand and generate human-like language. These models leverage deep learning techniques, particularly transformer 
                        architectures, to process and generate text across a wide range of contexts and tasks. LLMs are capable of performing diverse functions, 
                        including language translation, summarization, content creation, and even complex problem-solving. Their capabilities continue to expand 
                        as they are fine-tune for specific applications, making them invaluable tools in industries such as healthcare, education, customer 
                        support, and software development."""
                    }
                }
            ]
        )
    assert isinstance(exc_info.value, ValueError)

def test_executor_rewrite_valid():
    result = executor(
        **empty_args,
        action="rewrite",
        messages=[
            {
                "role":"human",
                "type":"text",
                "timestamp":"string",
                "payload":{
                    "text":"""Please, rewrite so like llms are these things that do like ai stuff and they like talk and write but not really like people 
                    but kinda, and they like use data or something, idk, like lots of data, and then they like learn, but not really learn like humans, 
                    just like, you know, math or whatever, and then they make stuff like words and answers, and ppl say they’re smart but they’re just 
                    like programs, and yeah, they’re everywhere now and ppl use them for like, idk, work or chatting or whatever"""
                }
            }
        ]
    )
    assert isinstance(result, str)

def test_executor_rewrite_invalid():
    with pytest.raises(ValueError) as exc_info:
        executor(
            **empty_args,
            action="rewrite123",
            messages=[
                {
                    "role":"human",
                    "type":"text",
                    "timestamp":"string",
                    "payload":{
                        "text":"""Please, rewrite so like llms are these things that do like ai stuff and they like talk and write but not really like people 
                        but kinda, and they like use data or something, idk, like lots of data, and then they like learn, but not really learn like humans, 
                        just like, you know, math or whatever, and then they make stuff like words and answers, and ppl say they’re smart but they’re just 
                        like programs, and yeah, they’re everywhere now and ppl use them for like, idk, work or chatting or whatever"""
                    }
                }
            ]
        )
    assert isinstance(exc_info.value, ValueError)

def test_executor_question_generation_valid():
    result = executor(
        **empty_args,
        action="question_generation",
        messages=[
            {
                "role":"human",
                "type":"text",
                "timestamp":"string",
                "payload":{
                    "text":"Please, create questions for Linear Algebra"
                }
            }
        ]
    )
    assert isinstance(result, dict)

def test_executor_question_generation_invalid():
    with pytest.raises(ValueError) as exc_info:
        executor(
            **empty_args,
            action="question_generation123",
            messages=[
                {
                    "role":"human",
                    "type":"text",
                    "timestamp":"string",
                    "payload":{
                        "text":"Please, create questions for Linear Algebra"
                    }
                }
            ]
        )
    assert isinstance(exc_info.value, ValueError)

def test_executor_custom_prompt_valid():
    result = executor(
        **empty_args,
        action="custom",
        messages=[
            {
                "role":"human",
                "type":"text",
                "timestamp":"string",
                "payload":{
                    "text":"Save this prompt: 'Please, create questions for Linear Algebra'"
                }
            }
        ]
    )
    assert isinstance(result, str)

def test_executor_custom_prompt_invalid():
    with pytest.raises(ValueError) as exc_info:
        executor(
            **empty_args,
            action="custom123",
            messages=[
                {
                    "role":"human",
                    "type":"text",
                    "timestamp":"string",
                    "payload":{
                        "text":"Save this prompt: 'Please, create questions for Linear Algebra'"
                    }
                }
            ]
        )
    assert isinstance(exc_info.value, ValueError)
