
test_questions = [
    {
        "question": "What are the types of agent memory?",
        "expected": "Sensory memory, short-term (working) memory, and long-term memory, with long-term further split into explicit (episodic and semantic) and implicit (procedural) memory.",
        "route": "vectorstore",
    },
    {
        "question": "Who won the last Super Bowl?",
        "expected": "The Seattle Seahawks defeated the New England Patriots 29-13 in Super Bowl LX.",
        "route": "web_search",
    },
    {
        "question": "Who won the last Cricket World Cup?",
        "expected": "Australia won the 2023 ODI Cricket World Cup, defeating India.",
        "route": "web_search",
    },
    {
        "question": "What is prompt engineering?",
        "expected": "The practice of designing and refining inputs (prompts) to language models to elicit more accurate, relevant, or useful outputs.",
        "route": "vectorstore",
    },
    {
        "question": "hi hello",
        "expected": None,
        "route": "direct_answer",
    },
]