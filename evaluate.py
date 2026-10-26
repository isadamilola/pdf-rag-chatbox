import os
import asyncio
import csv

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecision,
    AnswerRelevancy,
)

from rag import chunks, index, retrieve_chunks, answer_question

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Add GROQ_API_KEY to your .env file.")

# Groq through its OpenAI-compatible endpoint
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)

llm = llm_factory("openai/gpt-oss-20b", client=client, max_tokens=4096)

# Local embedding model, no paid API
embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

metrics = {
    "faithfulness": Faithfulness(llm=llm),
    "context_precision": ContextPrecision(llm=llm),
    "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
}

test_cases = [
    {
        "question": "What are the three guiding criteria for a medicine to qualify for this arrangement?",
        "reference": (
            "A very small patient group, an extremely severe condition, "
            "and considerable expected benefit from the medicine."
        ),
    },
    {
        "question": "What is the indicative global prevalence threshold for a very small patient group?",
        "reference": (
            "Fewer than approximately one patient per 100,000 inhabitants "
            "globally per medicine."
        ),
    },
    {
        "question": "What is the indicative minimum expected benefit compared with standard treatment?",
        "reference": (
            "A gain of at least approximately two good life years "
            "compared with standard treatment."
        ),
    },
    {
        "question": "How is the severity of the condition measured under the arrangement?",
        "reference": (
            "By absolute shortfall, with a minimum loss of approximately "
            "30 good life years."
        ),
    },
    {
        "question": "Who is responsible for price negotiations and tendering processes?",
        "reference": (
            "The Norwegian Hospital Procurement Trust, Division drug "
            "procurements (Sykehusinnkjøp HF Divisjon legemidler LIS)."
        ),
    },
    {
        "question": "What information should monitoring collect after a medicine is funded under the arrangement?",
        "reference": (
            "Information on treatment effect, safety, and the number of "
            "patients relevant for treatment with the medicine."
        ),
    },
]

async def score_with_retry(scorer, tries=4):
        for attempt in range(tries):
            try:
                return await scorer()
            except Exception as exc:
                if "429" in str(exc) and attempt < tries - 1:
                    await asyncio.sleep(15 * (attempt + 1))
                    continue
                raise

async def main():
    results = []

    for i, case in enumerate(test_cases, start=1):
        question = case["question"]
        print(f"\nEvaluating {i}/{len(test_cases)}: {question}")

        contexts = retrieve_chunks(question, chunks, index, top_k=3)
        response = answer_question(question, chunks, index)

        row = {
            "question": question,
            "reference": case["reference"],
            "response": response,
            "retrieved_contexts": " | ".join(contexts),
        }

        scorers = {
            "faithfulness": lambda: metrics["faithfulness"].ascore(
                user_input=question,
                response=response,
                retrieved_contexts=contexts,
            ),
            "context_precision": lambda: metrics["context_precision"].ascore(
                user_input=question,
                reference=case["reference"],
                retrieved_contexts=contexts,
            ),
            "answer_relevancy": lambda: metrics["answer_relevancy"].ascore(
                user_input=question,
                response=response,
            ),
        }

        for name, scorer in scorers.items():
            try:
                result = await score_with_retry (scorer)
                row[name] = result.value
                print(f"  {name}: {result.value:.3f}")
            except Exception as exc:
                row[name] = ""
                print(f"  {name} failed: {exc}")

        results.append(row)

    filename = "results_english.csv"
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    metric_columns = [
        "faithfulness",
        "context_precision",
        "answer_relevancy",
    ]

    failures = sum(
        1
        for row in results
        for metric in metric_columns
        if row.get(metric, "") == ""
    )

    total = len(results) * len(metric_columns)

    print(f"\nEvaluation finished: {total - failures}/{total} metric scores succeeded.")
    print(f"Failed metric scores: {failures}")
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    asyncio.run(main())