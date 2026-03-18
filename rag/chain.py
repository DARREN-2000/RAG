"""RAG chain construction using LangChain."""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """You are a helpful customer support assistant. Use ONLY the context
below to answer the question. If the answer is not contained in the context, say
"I'm sorry, I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(
    vector_store: FAISS,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
) -> RetrievalQA:
    """Build a Retrieval-QA chain backed by the given vector store.

    Args:
        vector_store: FAISS vector store containing embedded documents.
        model_name: OpenAI chat model to use for generation.
        temperature: Sampling temperature (0 = deterministic).
        k: Number of documents to retrieve for each query.

    Returns:
        A LangChain RetrievalQA chain.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain
