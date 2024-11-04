#%% md
# ---
#%% md
# # <span style="color:orange; font-weight:bold;">Regulatory AI Search Engine with Llama</span> <br>
# ### <span style="color:orange; font-weight:bold;">A Local Retrieval-Augmented Generation (RAG) </span> <br> <br>
# *created with ❤️ by Daniel Jacobowitz & Prasoon Singh*
#%% md
# ---
#%% md
# # **I. What tha fuck is a Retrieval-Augmented Generation (RAG)?**
#%% md
# Retrieval augmented generation (RAG) is a technique that combines information retrieval technique with language model generation to improve the accuracy and credibility of the generated text, and to better ground the model’s response in evidence. In RAG, a language model is augmented with an external knowledge base or a set of documents that is passed into the context window. <br>
# 
# At the core of RAG lies the seamless integration of two key components:
# 1. **Vector Databases/ Retriever:** These specialized databases store and index information in a vector representation (so called vector embedding), enabling similarity searches and retrieval of relevant data.
# 2. **Large Language Models (LLMs)/ Generator:** Takes the retrieved information and the original query as input, and produced human-like output that incorporates both factual knowledge and contextual understanding. <br>
# 
# By combining the retrieved information with the LLM`s language generation capabilities, the RAG system can produce more accurate, relevant and credible responses compared to using the LLM alone. <br>
# 
# To understand the concept better, let`s compare in the following the basic system with a RAG system. Let’s start with a basic system. A basic system is composed of a prompt, a model, and an output parser. The basic system illustrates the core idea of using LLMs in a structured way by defining the input (prompt), the processing step (model), and the output handling (output parser). It serves as a foundational concept that can be extended and combined with other components to build more complex chains and applications. Below, an illustration of the basic chain architecture is provided. <br> <br>
#  ![basic_chain.png](Basic_chain.png) <br> <br>
# 
# 
# A RAG system on the other hand adds context when responding to questions. Let’s break down RAG system by a RAG prompt, a RAG retriever, and a RAG chain as illustrated in the following. <br> <br>
#  ![basic_chain.png](RAG_chain.png) <br> <br>
# 
# Here`s a step-by-step breakdown how a RAG system is structured and how it works: <br>
# ***Data preparation:***
# 1. The raw data sources (e.g. PDF files) are loaded.
# 2. The information of the loaded data sources are extracted (e.g. text from the PDF files).
# 3. The huge amount of text is spilled/ chunked to smaller pieces.
# 4. The chunks are then indexed into a numerical vector  (Embedding) and stored as a vector database (vector store). <br> <br>
# ***RAG:*** <br>
# 1. A user provides a query or prompt to the RAG system.
# 2. The query is then converted to vector embeddings, so the information can be processed for the next step.
# 3. The retriever component searches through the created vector database to find the most relevant information related to the query (vector similarity search).
# 3. The retrieved relevant data/ information along with the original query, is passed to the generator component (the LLM).
# 4. The LLM generates a response that incorporates the retrieved information, providing a more accurate and context-aware answer to the user`s query.
# 
# By combining the strength of LLMs and information retrieval techniques, RAG systems offer a promising approach to overcome some of the limitations of LLMs and enhancing the quality and reliability of generated text. The benefits of using RAG in enhancing LLM credibility and accuracy are numerous:
# 1. **Improved factual accuracy:** By incorporating relevant information from reliable sources, RAG systems can help reduce the instances of factual errors of inconsistencies in the generated text.
# 2. **Enhanced credibility:** RAG systems can provide citations or references to the sources of information used in the generated response, increasing the credibility and trustworthiness of the output.
# 3. **Access to up-to-date information:** RAG systems can be designed to retrieve information from continuously updated sources, ensuring that the generated responses are based on the most recent and relevant information available.
# 4. **Customization and domain-specific knowledge:** RAG systems can be tailored to specific domains or use cases by incorporating domain-specific knowledge bases or documents, enabling the generation of more accurate and relevant responses within a particular field or context. <br>
# 
# **Useful links:**
# - https://medium.com/@dminhk/retrieval-augmented-generation-rag-explained-b1dd89979681
# 
# 
#%% md
# ---
#%% md
# # **II. Step-by-step instructions to build a RAG**
#%% md
# ---
#%% md
# ### **Step 1: Installing and loading the necessary Modules (incl. description)**
#%%
#!pip install langchain-community==0.2.4 langchain==0.2.3 faiss-cpu==1.8.0 unstructured==0.14.5 unstructured[pdf]==0.14.5 transformers==4.41.2 sentence-transformers==3.0.1

import pandas as pd

import os # --> interacts with the operating system, like creating files and directories, management of files and directories, input, output, environment variables etc.

#import pandas as pd # --> Analyses data and allows to manipulate data as dataframes

from langchain_community.llms import Ollama  # --> LLM classes provide access to the large language model (in this case locally to Ollama) APIs and services.

from langchain.document_loaders import UnstructuredFileLoader # --> Loads data from a source as Documents, which are a piece of text and associated metadata.
                                                              # --> In this case, the file loader uses the unstructured partition function and will automatically detect the file type.

from langchain.document_loaders.pdf import PyPDFDirectoryLoader # --> Loads data from a source as Documents, which are a piece of text and associated metadata.
                                                                # --> In this case, the file loader loads a directory with PDF files using pypdf and chunks at character level. The loader also stores page numbers in metadata.

from langchain_community.document_loaders import PyPDFLoader

from langchain.embeddings import HuggingFaceEmbeddings #--> Convert the text of the PDF files using Embeddings models into numerical vectors
#from langchain_community.embeddings.ollama import OllamaEmbeddings #--> Convert the text of the PDF files using Embeddings models into numerical vectors
#from langchain_ollama import OllamaEmbeddings #--> Convert the text of the PDF files using Embeddings models into numerical vectors

from langchain_community.vectorstores import FAISS # --> Creates a local vector store to store the created embeddings of the PDF files. LangChain has a number of built-in document transformers that make it easy to split, combine, filter, and otherwise manipulate documents.


from langchain.text_splitter import CharacterTextSplitter # --> Splits a long document (in our case the PDF files) into smaller chunks that can fit into the model`s context window. LangChain has a number of built-in document transformers that make it easy to split, combine, filter, and otherwise manipulate documents.
from langchain.text_splitter import RecursiveCharacterTextSplitter # --> Splits a long document (in our case the PDF files) into smaller chunks that can fit into the model`s context window. LangChain has a number of built-in document transformers that make it easy to split, combine, filter, and otherwise manipulate documents.
from langchain_experimental.text_splitter import SemanticChunker #--> Splits a long document (in our case the PDF files) into smaller chunks that can fit into the model`s context window. LangChain has a number of built-in document transformers that make it easy to split, combine, filter, and otherwise manipulate documents.

from langchain.chains import RetrievalQA # --> Chains refer to sequences of calls - whether to an LLM, a tool, or a data preprocessing step. This chain first does a retrieval step to fetch relevant documents, then passes those documents into an LLM to generate a response.

from langchain_core.prompts import ChatPromptTemplate # --> A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.
#from langchain.memory import ConversationBufferMemory # --> Creates a conversational memory that allows a LLM to remember previous interactions/ chats with the user. It helps to manage and store conversation history in a structured way and to maintain the context of a conversation over multiple interactions.

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler # --> LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. In this case, it is a callback for streaming.
                                                                                # --> Streaming: process of incrementally receiving and processing data generated by a Large Language Model (LLM) as it is produced, rather than waiting for the entire response to be generated before displaying it.
from langchain.callbacks.manager import CallbackManager # --> LangChain provides a callbacks system that allows you to hook into the various stages of your LLM application. In this case, it is to manage callbacks.

import streamlit as st # --> Streamlit is a free and open-source framework to rapidly build and share beautiful machine learning and data science web apps.

import time #This module provides various time-related functions. 

if not os.path.exists('C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/pdfFiles'): # Check if the folder does not exist
    os.makedirs('C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/pdfFiles') # create a folder in the specified path if it does not exist already

if not os.path.exists('C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/vectorDB'): # Check if the folder does not exist
    os.makedirs('C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/vectorDB') # create a folder in the specified path if it does not exist already

switch = 0
#%% md
# ---
#%% md
# ## **Step 2: Uploading the PDF files as text documents, chunking the documents and convert the documents to numerical vectors incl. indexation with vector embeddings and store them into a local database.** <br> <br>
#%%
# 1. Upload relevant PDF files
DATA_PATH = "C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/pdfFiles" # Directory where the relevant PDFs are stored

if switch == 1:
    def load_documents(DATA_PATH): # defines a function named 'load_documents' that takes one parameter, 'DATA_PATH'.
        document_loader = PyPDFDirectoryLoader(DATA_PATH) # creates the class 'PyPDFDirectoryLoader' designed to load PDF documents from a specified directory.
        return document_loader.load() # the load method on the document_loader instance and returns the result. The load method is expected to read and load the PDF documents from the directory specified by DATA_PATH. The load method loads data into Document objects.

    documents = load_documents(DATA_PATH) # the variable 'documents' is a object containing for every list elements a page of the PDF files.
    
    # --> Plausibility check
    print("\033[92mThe upload of the PDF files was successful!\033[0m")
    print('')
    print('CRR:')
    print (documents[0])
    print('')
    print('Banking Act:')
    print(documents[337])
    print('')
    print('MaRisk:')
    print(documents[600])
    print('')
#%%
# 2. Chunking the documents

if switch == 1:
    #text_splitter = SemanticChunker(embeddings=HuggingFaceEmbeddings(), breakpoint_threshold_type="percentile") # Apply the chunking based on cosine similarity. There a many breakpoint threshold types.
    #text_chunks = text_splitter.split_documents(documents) # #  splits the documents into chunks based on the similarity
    
    text_splitter = RecursiveCharacterTextSplitter( # function to split documents into chunks
                chunk_size=1500, # sets the maximum size of each chunk to 1500 characters.
                chunk_overlap=200, #  sets the overlap between consecutive chunks to 200 characters. Overlapping helps to ensure that context is preserved across chunks.
                length_function=len # specifies that the length of the chunks will be measured using the len function, which counts the number of characters.
            )

    text_chunks = text_splitter.split_documents(documents) #  splits the documents into chunks based on the specified chunk_size and chunk_overlap.
#%%
# 3. Convert the splitted text into Vector Embeddings incl. indexation and store the created vector data base locally

embeddings = HuggingFaceEmbeddings() # The used embeddings model is from HuggingFace
 #embeddings = OllamaEmbeddings(
    #        model = "llama3.2:3b",
    #  base_url = 'http://localhost:11434'
    #    )

if switch == 1:
   # Create and save the vector store
    knowledge_base = FAISS.from_documents(text_chunks, embeddings) # The created chunks are converted to numerical vectors and stored in a FAISS vector database.
    knowledge_base.save_local('C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/vectorDB') # The created FAISS vector database is stored locally in the given directory.

    print("\033[92mThe Embedding was successful!\033[0m")

#%% md
# ---
#%% md
# ## **Step 3: Declare the created data base as retriever and test the retrieval based on a cosine similarity search** <br> <br>
#%%
# 1. Declaration of the retriever based on the created database

retriever = FAISS.load_local( # This method converts the stored vector database into a retriever object. A retriever is typically used to search and retrieve relevant documents or information. In other words, it will be the base for the search as the retrieved information will then later be passed to the LLM.
    'C:/Users/49171/OneDrive/Desktop/Programieren/Python/PDF_test/Final/vectorDB', # Directory of the locally stored vector database
    embeddings, # Parameter to declare, that the embeddings should be loaded.
    allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k":5}) # Defines, that the 5 most similar text chunks should be output in the course of a cosine similarity search.
#%% md
# ---
#%% md
# ## **Step 4: Passing the retrieved information to the LLM** <br> <br>
#%%
# 1. Create a prompt template to pass the retrieved information and the original query in a standardised format to the LLM. 

# The input variable needs to be declared in the next steps. "Context" is the retrieved information and "question" the original query. 
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) # Create a ChatPromptTemplate object to pass it to the LLM.


#%%
# 2. Declare the preferred LLM to generate the answer for the query based on the retrieved information.

llm = Ollama( # The local ollama model is used for the generation of the answer.
    model="llama3.2:3b", # Model version
    temperature=0, # controls the randomness of the model's output. A higher temperature value (e.g., 1.5) makes the model's responses more random and creative, while a lower temperature value (e.g., 0.9) makes the responses more deterministic and focused. The default value is typically around 1.1. The chosen value of 0 means the model will produce the most deterministic and predictable output, essentially always choosing the highest probability token at each step. This makes the responses very repetitive and lacks the creativity or variability you'd see with higher temperatures. It's like playing it safe and not taking any chances.
    base_url = 'http://localhost:11434', # Local base URL
)
#%%
qa_chain = RetrievalQA.from_chain_type( # Creates a RetrievalQA object, which creates a connection ("chain") between the retrieved information and the declared LLM.
    llm, # Specifies the language model to be used for generating answers, which has been declared in the previous step.
    retriever=retriever, # Specifies the retriever to be used for fetching relevant documents or information from the stored vector database.
    chain_type_kwargs={ # A dictionary of additional keyword arguments for the chain type.
                "verbose": True, # Ensures that verbose mode is enabled within the chain.
                "prompt": prompt_template } # Specifies the prompt template to be passed to the LLM containing the retrieved information and the original query,
)
#%% md
# ---
#%% md
# ## **Step 5: Generate an answer from the LLM based on the retrieved information** <br> <br>
#%% md
# ---
#%% md
# # III. Appendix
#%% md
# ### **1. Code to generate answers for the validation**
#%% md
# ---
#%% md
# ### **2. Code to create a Web application for the RAG**
#%%
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/ING_Group_N.V._Logo.svg")
    st.title("Regulatory AI Search Engine")
    st.markdown('''
    ## About            
    This application is a regulatory chatbot using a local Retrieval-Augmented Generation (RAG)


    ''')
    st.markdown("---")
    st.markdown('''**Made with :orange_heart: by the Algorithm Alchemists**''')




def main():
    st.header("Chat with PDF :left_speech_bubble:")

    # Initialize chat history if not already in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add initial message from the assistant
        welcome_message = {"role": "assistant", "message": "Hi, how can I help you today?"}
        st.session_state.chat_history.append(welcome_message)

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["message"])
            if "Sources" in chat["message"]:
                with st.expander("🔍 Sources", expanded=False):
                    st.markdown(chat["message"])

    # User input
    if user_input := st.chat_input("Type here your question:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = qa_chain.invoke({"query": user_input})
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")

        assistant_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(assistant_message)

        # Display sources
        full_sources = []
        results = retriever.invoke(user_input)
        for doc in results:
            source_path = doc.metadata.get('source', 'No source available')
            source_filename = os.path.basename(source_path)
            filename_without_extension = os.path.splitext(source_filename)[0]
            source = f"Source: {filename_without_extension}"
            page = doc.metadata.get('page', 'No page available')
            if page != 'No page available':
                page_number = int(page) + 1
            else:
                page_number = "No page available"
            full_source = f"{source}, Page: {page_number}"
            full_sources.append(full_source)
        final_source = "\n".join(full_sources)
        source_message = {"role": "assistant", "message": final_source}
        st.session_state.chat_history.append(source_message)

        with st.expander("🔍 Sources", expanded=False):
            st.markdown(final_source)
    else:
        st.write("Please type your question regarding the CRR, MaRisk or the German Banking Act.")

if __name__ == "__main__":
    main()