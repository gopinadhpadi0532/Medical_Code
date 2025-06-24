__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

# --- Configuration ---
GUIDELINE_DB_PATH = "db/guideline_db"

# --- NEW ADVANCED PROMPT ---
# This prompt now expects two different types of context.
custom_prompt_template = """
You are an expert, meticulous AI Medical Coder. Your task is to analyze a user's question, the provided patient record, and the relevant official ICD-10 guidelines to determine the most accurate medical code.

**Your Inputs:**
1.  **Patient Record Context:** This contains the specific facts about the patient encounter.
2.  **Official Guideline Context:** This contains the rules and instructions from the ICD-10-CM guidelines that are relevant to the patient's condition.

**Your Task:**
Synthesize information from *both* contexts by strictly following the 10-step methodology. You must prioritize the rules from the Official Guideline Context when making your final decision. Present your answer in the required format with a clear explanation and step-by-step reasoning.

---
**PATIENT RECORD CONTEXT:**
{context}

---
**OFFICIAL GUIDELINE CONTEXT:**
{guideline_context}
---

**Chat History:**
{chat_history}

**Question:**
{question}

**Expert Medical Coder AI Final Answer:**
"""

CUSTOM_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "guideline_context", "chat_history", "question"]
)

# --- PDF and Text Processing Functions (No change) ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

# --- THE "TWO-BRAIN" CORE LOGIC ---
@st.experimental_singleton
def load_guideline_db():
    """Loads the persistent guideline vector database from disk."""
    if not os.path.exists(GUIDELINE_DB_PATH):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=GUIDELINE_DB_PATH, embedding_function=embeddings)

def get_session_vector_store(text_chunks):
    """Creates a temporary, in-memory vector store for the user's document."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_texts(texts=text_chunks, embedding=embeddings)


# This class replaces the simple ConversationalRetrievalChain with our advanced logic.
class TwoBrainQASystem:
    def __init__(self, patient_retriever, guideline_retriever, llm):
        self.patient_retriever = patient_retriever
        self.guideline_retriever = guideline_retriever
        self.llm = llm
        
        # Chain to condense a question and chat history into a standalone question
        self.question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        
        # Chain to answer a question based on provided context
        self.doc_chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=CUSTOM_PROMPT)
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def __call__(self, inputs):
        question = inputs["question"]
        chat_history_str = self.memory.load_memory_variables({})['chat_history']

        # 1. Create standalone question
        standalone_question = self.question_generator.run(chat_history=chat_history_str, question=question)

        # 2. Retrieve relevant docs from BOTH databases
        patient_docs = self.patient_retriever.get_relevant_documents(standalone_question)
        guideline_docs = self.guideline_retriever.get_relevant_documents(standalone_question)

        # 3. Generate the answer
        result = self.doc_chain(
            {
                "input_documents": patient_docs,
                "guideline_context": "\n".join([doc.page_content for doc in guideline_docs]),
                "question": standalone_question,
                "chat_history": chat_history_str
            },
            return_only_outputs=True
        )
        
        # 4. Update memory
        self.memory.save_context(inputs={"question": question}, outputs={"answer": result["output_text"]})
        
        return {"answer": result["output_text"], "chat_history": self.memory.chat_memory.messages}


def main():
    st.set_page_config(page_title="Expert AI Medical Coder", page_icon="⚕️")
    st.title("Expert AI-Powered Medical Coding Assistant 💬")

    # Load the permanent guideline database
    guideline_db = load_guideline_db()
    if guideline_db is None:
        st.error(f"Guideline database not found at '{GUIDELINE_DB_PATH}'. Please run `python create_guideline_db.py` first.")
        return

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("1. Provide Patient Record")
        pdf_docs = st.file_uploader("Upload PDF file(s):", accept_multiple_files=True, type="pdf")
        st.divider()
        manual_text = st.text_area("Or paste medical text here:", height=250)

        if st.button("2. Analyze Record"):
            if pdf_docs or manual_text:
                with st.spinner("Processing patient record..."):
                    raw_text = ""
                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs) + "\n"
                    if manual_text:
                        raw_text += manual_text
                    
                    text_chunks = get_text_chunks(raw_text)
                    patient_vector_store = get_session_vector_store(text_chunks)
                    
                    # Setup the advanced QA system
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, convert_system_message_to_human=True)
                    
                    st.session_state.conversation = TwoBrainQASystem(
                        patient_retriever=patient_vector_store.as_retriever(),
                        guideline_retriever=guideline_db.as_retriever(),
                        llm=llm
                    )
                    st.success("Analysis complete! Ready for questions.")
            else:
                st.error("Please provide a patient record to analyze.")

    st.header("3. Ask Coding Questions")

    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
             if i % 2 == 0:
                 with st.chat_message("user"):
                     st.write(message.content)
             else:
                 with st.chat_message("assistant"):
                     st.write(message.content)

    if user_question := st.chat_input("What is the code for...?"):
        if st.session_state.conversation:
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Consulting patient record and ICD-10 guidelines..."):
                    response = st.session_state.conversation({"question": user_question})
                    st.session_state.chat_history = response['chat_history']
                    st.write(response["answer"])
        else:
            st.warning("Please analyze a patient record first.")

if __name__ == '__main__':
    main()
