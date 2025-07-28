import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY not found. Please create a .env file and add your key.")
    sys.exit(1)

DB_FAISS_PATH = 'vectorstore/'

# Define the prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    """
    Creates a RetrievalQA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

def load_llm():
    """
    Loads the Google Generative AI model.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=google_api_key,
                                 temperature=0.7, convert_system_message_to_human=True)
    return llm

def qa_bot():
    """
    Initializes the QA bot.
    """
    if not os.path.exists(DB_FAISS_PATH):
        print(f"Error: Vector store not found at {DB_FAISS_PATH}")
        print("Please run 'python ingest.py' first to create the vector store.")
        sys.exit(1)
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def main():
    """
    Main function to run the chatbot.
    """
    chain = qa_bot()
    print("Chatbot is ready! Ask a question or type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        if user_input:
            result = chain.invoke({'query': user_input})
            print("Bot:", result["result"])
            # To see the source documents, you can uncomment the following lines:
            # print("\nSource Documents:")
            # for doc in result["source_documents"]:
            #     print(f"- {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()
