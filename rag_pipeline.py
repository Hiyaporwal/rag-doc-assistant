from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def load_qa_chain(db_path="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    prompt_template = """
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say "I don't know."

    Context: {context}
    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

def ask_question(chain, question):
    result = chain.invoke({"query": question})
    return result["result"], result["source_documents"]
