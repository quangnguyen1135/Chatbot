import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set Google API Key if not already set
if 'ChatOpenAi' not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA-MiyxXeCTQ0Nb8j86g67S65vTUaimZLk"

# Initialize chatbot components
llm = ChatGoogleGenerativeAI(model="gemini-pro", max_new_token=4064, temperature=0.5)

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", max_new_token=4064, temperature=0.1),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 8}, max_tokens_limit=4064),
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

def read_vectors_db():
    global vector_db_path
    vector_db_path = "vectorstores/db_faiss"
    embedding_model = GPT4AllEmbeddings()
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

db = read_vectors_db()

template = """system
Bạn là người cung cấp thông tin về công ty hay các dữ liệu cơ bản cho người nhân viên trong công ty.
Sử dụng thông tin sau đây để trả lời câu hỏi, hãy cung cấp đúng thông tin dữ liệu mà người nhân viên trong công ty đang cần, hỗ trợ người nhân viên trong công ty, hãy sử dụng prompt của gemini-pro để trả lời cho mượt
{context}\n
user\n{question}\n
assistant"""
prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, db)
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/bot_response', methods=['POST'])
def bot_response():
    user_question = request.json['question']
    prompt_context = "\n".join([f"- {utterance[0]}\n- {utterance[1]}" for utterance in conversation_history if utterance[1]])
    prompt_text = prompt.template.format(context=prompt_context, question=user_question)
    response = llm_chain.invoke({"query": prompt_text})
    conversation_history.append((user_question, response["result"]))
    return jsonify({'response': response["result"]})

if __name__ == '__main__':
    app.run(debug=True)
