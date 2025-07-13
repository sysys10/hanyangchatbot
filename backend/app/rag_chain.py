# app/rag_chain.py
# ──────────────────────────────────────────
#  한양위키 챗봇: VectorStore & 모델 로딩 전용
#  (프롬프트·QA 체인은 routes.py에서 구성)
# ──────────────────────────────────────────
import os, json
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from tqdm import tqdm

# 1) 환경 변수 로드 & 모델 준비
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(
    openai_api_key=api_key,
    model="text-embedding-3-large", # ✅ v3 대형 모델 명시
    dimensions=1024
)

llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4o"  # ✅ GPT-4o로 변경
)

# 2) 텍스트 분할기ß
splitter = TokenTextSplitter(
    encoding_name="cl100k_base",
    chunk_size=800,
    chunk_overlap=100
)

# 3) 문서 로더
def load_documents(folder: str = "data/AllWikiPages") -> list[Document]:
    docs: list[Document] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            rel   = os.path.relpath(fpath, folder).replace("\\", "/")
            label = os.path.splitext(fname)[0]

            parent = Document(page_content=text,
                              metadata={"source": rel, "label": label, "title": data.get("title", label)})
            docs.extend(splitter.split_documents([parent]))
    return docs

# 4) VectorStore 빌드/로드
def build_vectorstore(batch=100) -> FAISS:
    docs = load_documents()
    print(f"📄 총 {len(docs)}개의 문서를 임베딩합니다.")

    vs = None
    for i in tqdm(range(0, len(docs), batch), desc="🔄 임베딩 중"):
        part = docs[i:i+batch]
        tmp = FAISS.from_texts(
            texts=[d.page_content for d in part],
            embedding=embedding_model,
            metadatas=[d.metadata for d in part]
        )
        if vs is None:
            vs = tmp
        else:
            vs.merge_from(tmp)
    vs.save_local("vectorstore_index")
    print("✅ 임베딩 완료 및 저장됨: vectorstore_index/")
    return vs

def load_vectorstore() -> FAISS:
    if os.path.exists("vectorstore_index"):
        return FAISS.load_local("vectorstore_index",
                                embedding_model,
                                allow_dangerous_deserialization=True)
    return build_vectorstore()

# 5) 외부 공개 객체
vectorstore = load_vectorstore()
