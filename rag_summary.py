import os
import time
import redis
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from redis.commands.search.query import Query

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Các hằng số và cấu hình ---
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
REDIS_INDEX_NAME = "question_cache_idx_v3"

st.set_page_config(page_title="RAG + Semantic Cache", page_icon="🧠")

load_dotenv()
MY_GEMINI_API_KEY = os.getenv("MY_GEMINI_API_KEY")
if not MY_GEMINI_API_KEY:
    st.error("Vui lòng thiết lập biến môi trường MY_GEMINI_API_KEY.")
    st.stop()

try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
    r.ping()
except redis.exceptions.ConnectionError as e:
    st.error(f"Lỗi kết nối Redis: {e}. Hãy đảm bảo Redis server đang chạy.")
    st.stop()

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
embeddings = load_embedding_model()

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=MY_GEMINI_API_KEY,
    )
llm = load_llm()

def create_redis_index_alternative():
    try:
        r.ft(REDIS_INDEX_NAME).info()
    except redis.exceptions.ResponseError:
        st.sidebar.write(f"Đang tạo index '{REDIS_INDEX_NAME}'...")
        command = [
            "FT.CREATE", REDIS_INDEX_NAME, "ON", "HASH",
            "PREFIX", "1", "q:", "SCHEMA",
            "question", "TEXT", "short_answer", "TEXT",
            "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32", "DIM", EMBEDDING_DIM, "DISTANCE_METRIC", "COSINE"
        ]
        try:
            r.execute_command(*[str(item) for item in command])
        except Exception as e:
            st.error(f"Lỗi khi tạo index Redis: {e}")
            return False
    return True

INDEX_READY = create_redis_index_alternative()

def check_cache(question: str):
    if not INDEX_READY: return None
    try:
        query_vector_bytes = np.array(embeddings.embed_query(question), dtype=np.float32).tobytes()
        redis_query = (
            Query(f"*=>[KNN {TOP_K} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("short_answer", "score")
            .dialect(2)
        )
        res = r.ft(REDIS_INDEX_NAME).search(redis_query, {"vec": query_vector_bytes}).docs
        if not res: return None

        best_match = res[0]
        if float(best_match.score) <= SIMILARITY_THRESHOLD:
            st.sidebar.info(f"Cache Hit! Score: {float(best_match.score):.4f}")
            return best_match.short_answer

        st.sidebar.warning(f"Cache Miss. Gần nhất: {float(best_match.score):.4f}")
        return None
    except Exception as e:
        print(f"Lỗi khi kiểm tra cache: {e}")
        return None

def get_short_answer_from_context(retriever, question: str) -> str:
    template = "Dựa trên tài liệu sau, viết một bản tóm tắt súc tích chứa thông tin cần thiết để trả lời câu hỏi(6 từ tối đa).\nTài liệu: {context}\nCâu hỏi: {question}\nTóm tắt:"
    prompt = PromptTemplate.from_template(template)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(question)

def get_short_answer_from_general_knowledge(question: str) -> str:
    template = "Hãy tạo một bản tóm tắt ngắn gọn các thông tin trọng yếu để trả lời câu hỏi sau(6 từ tối đa).\nCâu hỏi: {question}\nTóm tắt:"
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

def get_final_answer(context: str, question: str) -> str:
    template = "Dựa vào thông tin sau, trả lời câu hỏi của người dùng một cách tự nhiên.\nThông tin: {context}\nCâu hỏi: {question}\nCâu trả lời:"
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def save_to_cache(question: str, short_answer: str):
    if not INDEX_READY:
        return
    try:
        embedding_vector = np.array(embeddings.embed_query(question), dtype=np.float32)
        vector_bytes = embedding_vector.tobytes()
        doc_id = f"q:{hash(question)}"
        r.hset(doc_id, mapping={
            b"question": question.encode("utf-8"),
            b"short_answer": short_answer.encode("utf-8"),
            b"embedding": vector_bytes
        })
    except Exception as e:
        st.error(f"Lỗi khi lưu vào cache: {e}")

# --- Giao diện Streamlit ---
st.title("Trợ lý AI với Semantic Cache")
st.caption("Mô phỏng hệ thống từ bài báo 'Semantic Caching of Contextual Summaries'")

if 'messages' not in st.session_state: st.session_state.messages = []
if 'retriever' not in st.session_state: st.session_state.retriever = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    st.header("Chế độ RAG")
    uploaded_file = st.file_uploader("Tải lên file PDF (tùy chọn)", type="pdf")
    if uploaded_file and st.button("Xử lý tài liệu"):
        with st.spinner("Đang xử lý PDF..."):
            with open("temp.pdf", "wb") as f: f.write(uploaded_file.read())
            docs = PyPDFLoader("temp.pdf").load()
            splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
            vectorstore = FAISS.from_documents(splits, embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        st.success("Tài liệu đã sẵn sàng!")

with st.sidebar.expander("Xem cache gần đây"):
    keys = r.keys("q:*")
    for k in keys[-5:]:
        st.markdown(f"- `{r.hget(k, 'question')}`")

# --- Luồng xử lý chính ---
if prompt := st.chat_input(" Nhập câu hỏi của bạn..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not INDEX_READY:
            st.error("Lỗi: Index của Redis không sẵn sàng.")
            st.stop()

        with st.spinner("Đang suy nghĩ..."):
            cached_content = check_cache(prompt)

            if cached_content:
                answer = get_final_answer(cached_content, prompt)
                st.info("Câu trả lời được tạo từ cache.", icon="⚡")
            else:
                st.warning("Không tìm thấy trong cache. Đang tạo mới...")
                if st.session_state.retriever is not None:
                    short_answer = get_short_answer_from_context(st.session_state.retriever, prompt)
                else:
                    short_answer = get_short_answer_from_general_knowledge(prompt)

                answer = get_final_answer(short_answer, prompt)
                save_to_cache(prompt, short_answer)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})