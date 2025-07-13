from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag_chain import vectorstore, embedding_model, llm
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import json, re, textwrap, traceback, requests, os
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from openai import OpenAI

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    followups: list[str] = []

def preprocess(q: str) -> str:
    q = re.sub(r"(\w+학과)\s*교수", r"\1 소속 교수", q.strip())
    return re.sub(r"\s+", " ", q)

def build_temp_faiss(filtered, embedding_model, batch_size=100):
    vs = None
    for i in range(0, len(filtered), batch_size):
        part = filtered[i:i+batch_size]
        tmp = FAISS.from_texts(
            texts=[d.page_content for d in part],
            embedding=embedding_model,
            metadatas=[d.metadata for d in part]
        )
        if vs is None:
            vs = tmp
        else:
            vs.merge_from(tmp)
    return vs

def hybrid_retriever(query: str, top_k: int = 4):
    keyword = query.split()[0]
    all_docs = list(vectorstore.docstore._dict.values())
    filtered = [d for d in all_docs if keyword in d.page_content]
    if filtered:
        temp_vs = build_temp_faiss(filtered, embedding_model, batch_size=100)
        return temp_vs.as_retriever(search_kwargs={"k": top_k})
    return vectorstore.as_retriever(search_kwargs={"k": top_k})

BLACKLIST_TERMS = [
    "죄송합니다. 해당 정보는 제공되지 않습니다.",
    "해당 주제에 대한 내용은 현재 확인할 수 없습니다.",
    "관련 정보를 찾을 수 없습니다. 학교 공식 채널에 문의해 주세요."
]

def extract_keywords(text: str) -> list[str]:
    return re.findall(r"[가-힣]{2,}", text)

def is_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    return SequenceMatcher(None, a, b).ratio() > threshold

def generate_followups(question: str, docs: list[Document]) -> list[str]:
    summaries = "\n\n".join(
        textwrap.shorten(d.page_content, width=600, placeholder=" ...") for d in docs[:2]
    )
    prompt = f"""
당신은 한양대학교 챗봇입니다.

[사용자 질문]
{question}

[참조 문서 요약]
{summaries}

위 정보를 바탕으로 사용자가 이어서 궁금해할 만한 후속 질문을 0~3개 한국어로 생성하세요. 
단, 질문의 대상이 되는 학과가 있다면 반드시 질문에 그 학과명을 포함하세요.

생성된 질문은 반드시 문서 내용을 기반으로 답변할 수 있어야 하며, 문서에 언급되지 않은 주제는 피하세요. 
다음과 같은 질문의 답변이 생성되면 안됩니다.: 
- "죄송합니다. 해당 정보는 제공되지 않습니다."
- "해당 주제에 대한 내용은 현재 확인할 수 없습니다."
- "관련 정보를 찾을 수 없습니다. 학교 공식 채널에 문의해 주세요."
출력 형식 예시: ["질문1", "질문2"]
"""
    resp = llm.invoke(prompt)
    try:
        followups = json.loads(resp.content)
        validated = []
        seen = set()
        for q in followups:
            q_clean = q.strip()

            if any(is_similar(q_clean, prev_q) for prev_q in seen):
                continue

            results = vectorstore.similarity_search(q_clean, k=1)
            if not results:
                continue

            content = results[0].page_content
            if any(term in content for term in BLACKLIST_TERMS):
                continue

            follow_terms = extract_keywords(q_clean)
            content_keywords = extract_keywords(content)
            matched = any(
                is_similar(f_term, c_term)
                for f_term in follow_terms
                for c_term in content_keywords
            )
            if matched:
                validated.append(q_clean)
                seen.add(q_clean)
            if len(validated) >= 3:
                break
        return validated
    except Exception as e:
        print("⚠️ follow-up JSON 파싱 또는 필터 실패:", e, "\n원문:", resp.content)
        return []

def search_web_tool(query: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get("https://www.google.com/search", params={"q": query}, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        print("웹에서 찾은 정보입니다.") # 웹 디버깅

        for g in soup.select("div.g")[:3]:
            title = g.select_one("h3")
            link = g.select_one("a")
            if title and link:
                results.append(f"{title.text.strip()} - {link['href']}")
        return "\n".join(results) if results else "웹 검색 결과가 충분하지 않습니다."
    except Exception as e:
        return f"웹 검색 중 오류 발생: {str(e)}"

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        query = preprocess(req.message)
        retriever = hybrid_retriever(query, top_k=4)

        prompt_template = PromptTemplate.from_template("""
당신은 한양대학교 챗봇입니다. 사용자의 질문에 주어진 문서 기반으로 정확하게 답변하세요.

문서에 유효한 정보가 있으면 해당 정보만 바탕으로 답변하세요.

만약 문서에서 관련 정보를 전혀 찾을 수 없다면, 반드시 아래 세 문장 중 하나만 그대로 답변하세요:
- "죄송합니다. 해당 정보는 제공되지 않습니다."
- "해당 주제에 대한 내용은 현재 확인할 수 없습니다."
- "관련 정보를 찾을 수 없습니다. 학교 공식 채널에 문의해 주세요."

위 세 문장은 변경하거나 재구성하지 마세요.

질문: {question}
문서 내용:
{summaries}

답변:
""")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": prompt_template,
                "document_variable_name": "summaries"
            },
            return_source_documents=True
        )

        qa_result = qa_chain({"query": query})
        answer = qa_result.get("result", "죄송합니다. 정보를 찾을 수 없습니다.")
        sources = qa_result.get("source_documents", [])
        followups = generate_followups(query, sources)

        if answer.strip().strip('"').strip("'") in BLACKLIST_TERMS:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "웹 검색을 통해 정보를 찾습니다.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "검색할 질문"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]

            messages = [
                {"role": "system", "content": "당신은 한양대학교 챗봇입니다."},
                {"role": "user", "content": query}
            ]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "search_web"}}
            )

            message = response.choices[0].message

            if message.tool_calls:
                tool_call = message.tool_calls[0]
                tool_args = json.loads(tool_call.function.arguments)
                web_result = search_web_tool(tool_args["query"])

                messages.append(message)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name, 
                    "content": web_result
                })
                
                messages.append({
                    "role": "system",
                    "content": "웹에서 찾은 정보를 바탕으로 간결하고 명확하게 답변하세요. 사용자가 만족할 수 있도록 최신 정보를 전달하세요. ‘정보가 부족하다’는 표현은 사용하지 마세요."
                })

                second_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )

                answer = second_response.choices[0].message.content.strip()
        
        print("\n" + "=" * 40)
        print(f"🗣 질문: {query}")
        for i, d in enumerate(sources):
            meta = d.metadata
            print(f"\n— 문서 {i+1} — label:{meta.get('label')} source:{meta.get('source')}")
        print("\n💬 최종 출력:", answer)
        print("=" * 40 + "\n")

        return ChatResponse(response=answer, followups=followups)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))