# translation_chains.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Translation chain factory
def get_translation_chain(target_language_name: str):
    translation_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a translation assistant. Translate the user's input to {target_language_name} with the highest possible quality.
            If feedback is provided, revise and improve your previous translation.
            Return only the final translated {target_language_name} text — do not include any explanations, notes, or formatting."""
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return translation_prompt | llm

# Static reflection chain
reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a translation reviewer grading the translations. Generate critique and recommendations for the user's inputs in terms of accuracy.
        Always provide detailed recommendations, including requests for cultural appropriateness, virality, style, etc."""
    ),
    MessagesPlaceholder(variable_name="messages")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
reflect_chain = reflection_prompt | llm
