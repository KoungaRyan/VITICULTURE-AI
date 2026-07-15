from agent.vigne_agent import build_vigne_agent  # nom du graphe compilé

graph = build_vigne_agent()

result = graph.invoke(
    {"messages": [("user", "Quel est l'état de la parcelle irouleguy-pilote-01 ?")]},
    config={"configurable": {"thread_id": "test-data-commons"}},
)
print(result)