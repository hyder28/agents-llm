# translation_reflect_agent.py

from typing import List, Sequence
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from translation_chains import get_translation_chain, reflect_chain

TRANSLATE = "translate"
REFLECT = "reflect"

LANGUAGE_NAME_MAP = {
    "ta": "Tamil",
    "zh-CN": "Simplified Chinese",
    "es-ES": "Spanish"
}

def build_graph(target_language_name: str):
    translate_chain = get_translation_chain(target_language_name)

    def translation_node(state: Sequence[BaseMessage]):
        return translate_chain.invoke({"messages": state})

    def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        res = reflect_chain.invoke({"messages": messages})
        return [HumanMessage(content=res.content)]

    def should_continue(state: List[BaseMessage]):
        return END if len(state) > 4 else REFLECT

    builder = MessageGraph()
    builder.add_node(TRANSLATE, translation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.set_entry_point(TRANSLATE)
    builder.add_conditional_edges(TRANSLATE, should_continue, {END: END, REFLECT: REFLECT})
    builder.add_edge(REFLECT, TRANSLATE)

    return builder.compile()

if __name__ == "__main__":
    input_text = "Hyperliquid is doing >$100M daily volume AI agents surged past a $7.6B market size in 2025"

    for target_lang in ["es-ES", "zh-CN", "ta"]:
        target_language_name = LANGUAGE_NAME_MAP[target_lang]

        graph = build_graph(target_language_name)

        with open("translation_agent.png", "wb") as f:
            png_data = graph.get_graph().draw_mermaid_png()
            f.write(png_data)
        print("Graph saved as 'translation_agent.png'")

        inputs = HumanMessage(content=input_text)
        response = graph.invoke(inputs)[-1]
        print(f"Final Translation ({target_lang}): {response.content}")
