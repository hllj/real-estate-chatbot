from langgraph.graph import StateGraph, END
from src.graph.state import GraphState
from src.graph.nodes import extract_info, predict_price, chatbot

def create_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("extract_info", extract_info)
    workflow.add_node("predict_price", predict_price)
    workflow.add_node("chatbot", chatbot)

    # Set entry point
    workflow.set_entry_point("extract_info")

    # Define edges
    # extract -> predict -> chatbot -> END
    workflow.add_edge("extract_info", "predict_price")
    workflow.add_edge("predict_price", "chatbot")
    workflow.add_edge("chatbot", END)

    return workflow.compile()

if __name__ == "__main__":
    graph = create_graph()
    print(graph.get_graph().draw_mermaid())
    print("Workflow graph created and visualized.")