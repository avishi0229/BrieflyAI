#Actionableitems , decision , questions 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_groq_llm, invoke_llm_chain


def get_llm():
    return get_groq_llm(temperature=0.2)



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_meeting_insights(transcript: str) -> dict:
    chain = build_chain(
        "You are an expert meeting analyst. Extract information from the meeting transcript.\n"
        "Return exactly these three sections with the headings shown:\n\n"
        "ACTION ITEMS\n"
        "- Task description | Owner | Deadline (or 'Not specified')\n"
        "If none, write 'No action items found.'\n\n"
        "KEY DECISIONS\n"
        "- One decision per line\n"
        "If none, write 'No key decisions found.'\n\n"
        "OPEN QUESTIONS\n"
        "- One unresolved question or follow-up topic per line\n"
        "If none, write 'No open questions found.'"
    )

    response = invoke_llm_chain(chain, transcript[:12000])
    sections = {
        "action_items": "No action items found.",
        "key_decisions": "No key decisions found.",
        "open_questions": "No open questions found.",
    }
    current_section = None
    section_names = {
        "ACTION ITEMS": "action_items",
        "KEY DECISIONS": "key_decisions",
        "OPEN QUESTIONS": "open_questions",
    }
    lines = {
        "action_items": [],
        "key_decisions": [],
        "open_questions": [],
    }
    for line in response.splitlines():
        heading = line.strip().upper().rstrip(":")
        if heading in section_names:
            current_section = section_names[heading]
        elif current_section in lines and line.strip():
            lines[current_section].append(line.strip())

    for name, section_lines in lines.items():
        if section_lines:
            sections[name] = "\n".join(section_lines)
    return sections


def extract_action_items(transcript:str)->str:
    return extract_meeting_insights(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return extract_meeting_insights(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return extract_meeting_insights(transcript)["open_questions"]