import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from flask import Flask, render_template, request, jsonify
import markdown2

# Initialize Flask app
app = Flask(__name__)

# Set Google API Key if not already set
if 'ChatOpenAi' not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyA-MiyxXeCTQ0Nb8j86g67S65vTUaimZLk"

# Initialize chatbot components
llm = ChatGoogleGenerativeAI(model="gemini-pro", max_new_token=2048, temperature=0.5)

def create_prompt(template):
  """
  Creates a PromptTemplate with support for basic formatting.

  Args:
      template: The template string containing potential formatting markers.

  Returns:
      A PromptTemplate object.
  """
  # Identify lines starting with "#" as headings (H1, H2, etc.)
  formatted_template = []
  current_heading_level = 0
  for line in template.splitlines():
    if line.startswith("#"):
      heading_level = len(line.lstrip("#"))
      formatted_template.append(f"##{heading_level} {line[heading_level:]}")
      current_heading_level = heading_level
    else:
      # Indent list items based on current heading level
      if current_heading_level > 0:
        formatted_template.append("  " * current_heading_level + line)
      else:
        formatted_template.append(line)

  prompt = PromptTemplate(template="\n".join(formatted_template), input_variables=["context", "question"])
  return prompt

def create_qa_chain(prompt, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", max_new_token=2048, temperature=0.5),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 8}, max_tokens_limit=2048),
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

template = """## Question-Answering System
**System:**

- **Information Retrieval:** Provides relevant information from the database based on user queries.
- **Interactive Dialogue:** Engages in natural language conversations with users to understand their requests and provide accurate responses.
- **Data-Driven Answers:** Retrieves information directly from the database, ensuring factual accuracy and avoiding content creation or speculation.
- **Knowledge Base Expansion:** Continuously learns and updates its knowledge base to improve its ability to answer diverse questions.

**User:**

- **Question Formulation:** Clearly expresses their questions in natural language, using keywords and context to convey their intent.
- **Interactive Communication:** Engages in a back-and-forth dialogue with the system to refine their questions and receive clarification if needed.
- **Feedback and Evaluation:** Provides feedback on the system's responses to help improve its performance and accuracy.

**Assistant:**
- **Natural Language Processing:** Understands and interprets user queries using advanced NLP techniques.
- **Context Awareness:** Considers the context of the conversation and user's previous interactions to provide relevant and consistent responses.
- **Database Access:** Efficiently retrieves and processes information from the database to answer user questions accurately and promptly.
- **Response Generation:** Generates clear, concise, and informative responses in natural language, tailored to the specific needs of the user, tương tác với người dùng.

**Prompt:**

{context}\n
**User:** {question}\n

"""

prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, db)
conversation_history = []

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat')
def chat():
    return render_template('index.html')

@app.route('/bot_response', methods=['POST'])
def bot_response():
    user_question = request.json['question']
    prompt_context = "\n".join([f"- {utterance[0]}\n- {utterance[1]}" for utterance in conversation_history if utterance[1]])
    prompt_text = prompt.template.format(context=prompt_context, question=user_question)
    response = llm_chain.invoke({"query": prompt_text})

    # Chuyển đổi văn bản từ Markdown sang HTML
    html_response = markdown2.markdown(response["result"])

    conversation_history.append((user_question, response["result"]))
    return jsonify({'response': html_response})

if __name__ == '__main__':
    app.run(debug=True)
