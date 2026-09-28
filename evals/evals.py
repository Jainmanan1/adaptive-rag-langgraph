 
from AdaptiveRag import run_pipeline
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextPrecision, Faithfulness,AnswerRelevancy,ContextRecall
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.run_config import RunConfig
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# cd D:\python\crag                          
#  python -m evals.evals  



run_config = RunConfig(
    timeout=800,
    max_retries=2,
    max_workers=1,
)

questions = [
    {
        "question": "What is prompt engineering?",
        "reference": (
            "Prompt engineering is the practice of designing and "
            "optimizing prompts to guide an AI model toward desired outputs."
        ),
    },
    # {
    #     "question": "What is generative AI?",
    #     "reference": (
    #         "Generative AI is a type of artificial intelligence that "
    #         "can generate new content such as text, images, audio, or code."
    #     ),
    # },
    # {
    #     "question": "What is retrieval augmented generation?",
    #     "reference": (
    #         "Retrieval-augmented generation combines information retrieval "
    #         "with language generation by retrieving relevant external "
    #         "information and using it to generate an answer."
    #     ),
    # },
]


samples =[]

for item in questions:
    question = item["question"]
    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    result = run_pipeline(question)
    answer = result.get("generation","")
    documents = result.get("documents", [])
    contexts = [doc.page_content for doc in documents]
    sample = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "reference": item["reference"],
    }

    samples.append(sample)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED DOCUMENTS:", len(contexts))  
    print("CONTEXTS PREVIEW:", [c[:100] for c in contexts])  
#dataset

dataset = EvaluationDataset.from_list(samples)

result = evaluate(
    dataset=dataset,
    metrics=[

    AnswerRelevancy(),ContextPrecision(),ContextRecall()
    ],
    llm=llm,
    embeddings=embeddings,
    batch_size=1,
    run_config=run_config,
    raise_exceptions=True,

)


print("\n")
print("=" * 60)
print("RAGAS RESULTS")
print("=" * 60)

print(result)
print(result.to_pandas())