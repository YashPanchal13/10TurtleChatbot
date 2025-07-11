import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# ==== Set API Key ====
os.environ["GOOGLE_API_KEY"] = "AIzaSyB2OrRw0vcxVaZetYmB6BsOLkhcAiGxNTM"

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ChatRequest(BaseModel):
    question: str

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local("_faiss_10turtle_index_", embedding_model, allow_dangerous_deserialization=True)

prompt_template = PromptTemplate.from_template("""
You are a smart, friendly, and helpful AI assistant for a company that specializes in there provided services.

Your job is to assist users by answering their questions using only the information in the context provided below. If the context includes related case studies (also referred to as "work" or "projects"), highlight them as examples of successful implementations.

Communication Style:
- Sound confident and helpful, like you're part of the team.
- Always keep a positive tone.
- If an exact match to the user’s request isn’t found, suggest the closest relevant services in a helpful and optimistic way.
- Do not use phrases like "we don’t offer," "not listed," or "isn't available."
- Do not start with negative phrasing.
- Avoid mentioning "context" or "provided text" — just speak naturally.
                                            
-----

User Question:
{question}

Available Context:
{context}

Instructions:
- Use only what’s explicitly found in the context — do not guess or fabricate services or projects.
- If a case studies (also called work/project) is found that matches the user’s interest, mention it as a strong example.
- Emphasize the most relevant services and case studies.
- Seamlessly embed any available links next to their matching names.
""")
# - If a source link is available, include it inline using Markdown format like `[Service Name](https://...)` or `[Case Study Name](https://...)`.

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
from langchain.schema import HumanMessage, AIMessage
memory.chat_memory.messages = [
    AIMessage(content="Hello! I'm your assistant. How can I help you today?")
]

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatGoogleGenerativeAI(model="models/gemini-2.0-flash"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    memory=memory,
    return_source_documents=True,
    # combine_docs_chain_kwargs={"prompt": prompt_template}
)

def generate_followup_questions(question, answer):
    prompt = f"""
    Based on the following question and answer, generate 3 small follow-up questions:
    Q: {question}
    A: {answer}
    Format each question as a separate line.
    """
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")
    response = llm.invoke(prompt)
    content = getattr(response, "content", str(response))
    return [line.strip("-• ").strip() for line in content.strip().split("\n") if line.strip()]

def extract_title_from_work_url(url):
    try:
        raw_title = url.split("/work/")[1].split("?")[0]
        title = raw_title.replace("-", " ").replace("&", "and")
        return re.sub(r'\s+', ' ', title).strip()
    except:
        return url

def is_greeting(message: str) -> bool:
    greetings = ["hi", "hello", "hey", "how are you", "good morning", "good afternoon", "good evening"]
    return any(greet in message.lower() for greet in greetings)
  
def convert_answer_to_html(text):
    paragraphs = re.split(r'\n\s*\n|(?<=\.)\s{2,}', text.strip())
    html_paragraphs = []
    for para in paragraphs:
        para = re.sub(r'(\$\d+)', r'<strong>\1</strong>', para)
        html_paragraphs.append(f"<p>{para.strip()}</p>")
    return "\n".join(html_paragraphs)

# === Simulated work links ===
work_links = ['https://10turtle.com/work/La-casa--de-las-capsulas?id=67fd06606af1dd50d0411feb',
 'https://10turtle.com/work/Rare-&-Remarkable?id=681b271c45155d582933ad24',
 'https://10turtle.com/work/Amor----?id=67f36b5f2a0c6d9d73acbeb7',
 'https://10turtle.com/work/Faith-In-Fashion?id=680c7a94e43dd9720d65ad3e',
 'https://10turtle.com/work/Imalloch-mcclean-+more?id=680b79112b28d873fb686335',
 'https://10turtle.com/work/Tomtoms-Burritos?id=67f39ff32a0c6d9d73accd72',
 'https://10turtle.com/work/The-Sugar-Papi?id=68134dd530a164f89630377b',
 'https://10turtle.com/work/ComfoFeet?id=67f0ca0f2a0c6d9d73acafb8',
 'https://10turtle.com/work/TYH-Warehousing?id=67f8998a6af1dd50d040ff4e',
 'https://10turtle.com/work/Mex-Insurance?id=68187568f713e21ad5cbe87e',
 'https://10turtle.com/work/Splash-In-Style?id=680c5bc7e43dd9720d65aa78',
 'https://10turtle.com/work/Cyberowl?id=680f567fe43dd9720d65b8ed',
 'https://10turtle.com/work/-77meridian?id=67f0ba202a0c6d9d73acace8',
 'https://10turtle.com/work/Steamysips?id=67973cce6ce7cabdab474bc0',
 'https://10turtle.com/work/ParleG?id=6797364c6ce7cabdab474bbc',
 'https://10turtle.com/work/Monkey-Tails?id=679761096ce7cabdab474bc9',
 'https://10turtle.com/work/ProXCath-Medical?id=67efd3c82a0c6d9d73aca915',
 'https://10turtle.com/work/Beauty-tail?id=679767d96ce7cabdab474bcf',
 'https://10turtle.com/work/Yoga-Mind-Li?id=67ecddc264bf8a6d6c392bff',
 'https://10turtle.com/work/Logo-AI?id=676f868c06ad7bb36c3e2922',
 'https://10turtle.com/work/PRO-Mate-Algorithmic-Trading-Dashboard?id=677381f06eea479324e68139',
 'https://10turtle.com/work/Wirebewegen?id=67975c776ce7cabdab474bc5',
 'https://10turtle.com/work/Torn-Ranch?id=67efbf8064bf8a6d6c393834',
 'https://10turtle.com/work/Topbright?id=679767ff6ce7cabdab474bd0',
 'https://10turtle.com/work/Houston-Mind-&-Brain?id=677283c1ac4f73637162354f',
 'https://10turtle.com/work/Exhibitive?id=6773630a6eea479324e67d22',
 'https://10turtle.com/work/Mobile-as-a-Service?id=677363ef6eea479324e67d33',
 'https://10turtle.com/work/Rawad-Dalal?id=67a70e3409bf31ac0d5de577',
 'https://10turtle.com/work/Classic-Garage?id=6772732bac4f73637162330f',
 'https://10turtle.com/work/Ultra?id=67737c786eea479324e68096',
 'https://10turtle.com/work/Mad-Kanna?id=6772727fac4f7363716232f8',
 'https://10turtle.com/work/Iflytek?id=677211a5b482de2ee0e90ae6',
 'https://10turtle.com/work/Narke-Hydraulik?id=677274a9ac4f736371623331',
 'https://10turtle.com/work/Yodidyouknow?id=677369e66eea479324e67dd4',
 'https://10turtle.com/work/Stem-Discovery-Boxes-----------------------------------------------?id=6773668d6eea479324e67d70',
 'https://10turtle.com/work/Black-Tie-Carts?id=67728258ac4f7363716234fd',
 'https://10turtle.com/work/Steezylnk?id=677365ca6eea479324e67d51',
 'https://10turtle.com/work/Truelynn.co?id=677213a6b482de2ee0e90afb',
 'https://10turtle.com/work/Candle-Corture?id=679764136ce7cabdab474bca',
 "https://10turtle.com/work/It's-Totwotoo?id=6772756aac4f736371623340",
 'https://10turtle.com/work/Beawhale?id=677367e76eea479324e67db5',
 'https://10turtle.com/work/Lime-grace?id=67974e206ce7cabdab474bc3',
 'https://10turtle.com/work/Escaped-The-9-to-5?id=67727133ac4f7363716232db',
 'https://10turtle.com/work/Lumora?id=67921cbc0ac8b3cd9cb70a7c',
 'https://10turtle.com/work/Eco4?id=679764956ce7cabdab474bcb',
 'https://10turtle.com/work/Taste-My-City?id=67737f5c6eea479324e680ea',
 'https://10turtle.com/work/Crypto-Trading?id=677379b76eea479324e68009',
 'https://10turtle.com/work/Calico?id=67921b410ac8b3cd9cb70a78',
 'https://10turtle.com/work/Jemma-russo?id=679766d66ce7cabdab474bcc',
 'https://10turtle.com/work/Issac-Advice?id=679740ac6ce7cabdab474bc1',
 'https://10turtle.com/work/VoiceGen-AI?id=67720dbd396c42a095965499',
 'https://10turtle.com/work/House-AI?id=67720fbb396c42a0959654ac',
 'https://10turtle.com/work/Bezoz-Estate?id=6798657ec9682c41184427a5',
 'https://10turtle.com/work/Taylor-Sterling?id=67921cf80ac8b3cd9cb70a7e',
 'https://10turtle.com/work/Schaufelle?id=679767256ce7cabdab474bcd',
 'https://10turtle.com/work/Dall:-AI-Image-&-Art-Generator?id=676e5ab472ca87f78b1d21d2',
 'https://10turtle.com/work/Fueling-The-Future?id=679870f6c9682c41184427a6',
 'https://10turtle.com/work/Wibes?id=67975ce86ce7cabdab474bc6',
 'https://10turtle.com/work/Casio?id=6797678c6ce7cabdab474bce',
 'https://10turtle.com/work/Ichi-Ichi?id=67975a3b6ce7cabdab474bc4',
 'https://10turtle.com/work/Ask-GPT?id=67720c06396c42a09596545b',
 'https://10turtle.com/work/Harder-Day?id=68563abf0756190f7b6549a6',
 'https://10turtle.com/work/Eleven-International?id=685545470756190f7b64a64c',
 'https://10turtle.com/work/MANA?id=6846d7802e664fcc8d12c84c',
 'https://10turtle.com/work/Vital-tropics?id=6843f2e99269878a01238ef1',
 'https://10turtle.com/work/MOVEMENT-MORTGAGE?id=6846a84b9269878a01254eef',
 'https://10turtle.com/work/Statement?id=685661d26ab7fab6eda4c2a1',
 'https://10turtle.com/work/CREATIVE-EVENT-DESIGN?id=6846b83e2e664fcc8d126f26',
 'https://10turtle.com/work/Trust-Convenience?id=6843df9a9269878a01236eb0',
 'https://10turtle.com/work/Caffeine-Crashers?id=6843d04f9269878a01235b56',
 'https://10turtle.com/work/FinalRound-AI?id=684674759269878a0124b912',
 'https://10turtle.com/work/PITERO-STEENBEEK?id=6846d0892e664fcc8d12b57d',
 'https://10turtle.com/work/EXP-Luxury?id=6836a0085748d4b165c3c6c0',
 'https://10turtle.com/work/Lucid-Awaken-Your-Senses?id=6843c3999269878a01234b98',
 'https://10turtle.com/work/KendiKids-Upendo?id=68315826e988cf52f82d0b0c',
 'https://10turtle.com/work/Onshii?id=68358c1c9f91917ebdcf245c',
 'https://10turtle.com/work/Marvel-Sports?id=683467f1e988cf52f82e5b6d',
 'https://10turtle.com/work/Mana?id=683d39a4924116bf26dcf9db',
 'https://10turtle.com/work/EXP-Luxury?id=6836b9525748d4b165c3d64a',
 'https://10turtle.com/work/Prishna?id=68315e5de988cf52f82d0fc4',
 'https://10turtle.com/work/Mana?id=68340a44e988cf52f82e37f0',
 'https://10turtle.com/work/Hilgard?id=6836b2dc5748d4b165c3d322',
 'https://10turtle.com/work/Tribal-Patterns?id=68359c439f91917ebdcf58f5',
 'https://10turtle.com/work/M-&-A?id=6836acc75748d4b165c3cd40',
 'https://10turtle.com/work/El-Fogón-Menu?id=6815b49530a164f89630472c',
 'https://10turtle.com/work/Cuppa?id=681a08f9bc5752a24887a223',
 'https://10turtle.com/work/ETF-Group?id=681b31f345155d582933cef9',
 'https://10turtle.com/work/Geometry?id=682476429765e8198179612a',
 'https://10turtle.com/work/Femmelife?id=682c773805cee7a744954a03',
 'https://10turtle.com/work/EV+-HIPOTECUS?id=682dcad3e988cf52f82a67ac',
 'https://10turtle.com/work/Patria-Lending?id=6818876bf713e21ad5cc2068',
 'https://10turtle.com/work/WALL-DESIGN?id=6815c01b2495377fee431c59',
 'https://10turtle.com/work/La-2e-Classe?id=6819a18628a45258ae885f45',
 'https://10turtle.com/work/Epicbend-EpicFedrick?id=682ea935e988cf52f82b117d',
 'https://10turtle.com/work/Zabrina-Cox?id=681b51b445155d582934395d',
 'https://10turtle.com/work/Ajanta-Fire-Work?id=6818ad8628a45258ae882277',
 'https://10turtle.com/work/IQUNA?id=6810ba4c1d6edcbff94d5d49',
 'https://10turtle.com/work/Playglobal.game?id=680f64e2e43dd9720d65ba7f',
 'https://10turtle.com/work/Prosprity-Finance-Group?id=6811cc001d6edcbff94d66c7',
 'https://10turtle.com/work/Pleasant-Paws-Pet-Center?id=6810c9e31d6edcbff94d618e',
 'https://10turtle.com/work/CureSHANK?id=6811e6131d6edcbff94d6b63',
 'https://10turtle.com/work/Grumpy-Alpha-Boss?id=6810c81b1d6edcbff94d60ed',
 'https://10turtle.com/work/REDITUS?id=681340a530a164f896303558',
 'https://10turtle.com/work/PKAO60*35?id=681097731d6edcbff94d586c',
 'https://10turtle.com/work/THE-GRIND?id=6814aeb130a164f896304062',
 'https://10turtle.com/work/Preshifter?id=680f71ac1d6edcbff94d54ce',
 'https://10turtle.com/work/Tillian-Simmons?id=680f456be43dd9720d65b725',
 'https://10turtle.com/work/Myspy?id=68120c661d6edcbff94d6de0',
 'https://10turtle.com/work/Powering?id=680b82552b28d873fb686514',
 'https://10turtle.com/work/la2eclasse?id=67fcb6b46af1dd50d0411aa0',
 'https://10turtle.com/work/Uxbridge-Vets?id=68078cc82b28d873fb685711',
 'https://10turtle.com/work/Fuel-Junkie?id=680c8a5de43dd9720d65b068',
 'https://10turtle.com/work/Astreya?id=680f1d7ee43dd9720d65b478',
 'https://10turtle.com/work/Le-Collectif’s?id=680c64e7e43dd9720d65abfb',
 'https://10turtle.com/work/Promille?id=680c7d88e43dd9720d65aebc',
 "https://10turtle.com/work/Children's-Guide?id=6805e68a168580c9774c07ec",
 'https://10turtle.com/work/Jewelry-Sales-Academy?id=6808cad92b28d873fb685baf',
 'https://10turtle.com/work/Allore-Studio?id=67fcb90a6af1dd50d0411bc9',
 'https://10turtle.com/work/Mentes-Expertas?id=6808bdf62b28d873fb685a48',
 'https://10turtle.com/work/Fashion?id=680b68872b28d873fb686114',
 'https://10turtle.com/work/Asmblr?id=67f61e47f3cdebff69681bbc',
 'https://10turtle.com/work/My-Leckerly-Paradiesly?id=67f63335d919fee9b7e3fbf1',
 'https://10turtle.com/work/FotoFreude?id=67fa08ea6af1dd50d0410e9a',
 'https://10turtle.com/work/Rose-Wallis-Atelier?id=67fa10996af1dd50d0410f85',
 'https://10turtle.com/work/GlobalGrad-Ireland?id=67f8dcee6af1dd50d0410339',
 'https://10turtle.com/work/Priceless-Education?id=67fc90cb6af1dd50d04113a3',
 'https://10turtle.com/work/Yu-Nutrition?id=67f617aaf3cdebff69681916',
 'https://10turtle.com/work/SocreateZ?id=67f908556af1dd50d04107b0',
 'https://10turtle.com/work/Dee’s-Organics?id=67f9fc4f6af1dd50d0410c35',
 'https://10turtle.com/work/Dar-Pilates-Sharjah?id=67fa15186af1dd50d041105c',
 'https://10turtle.com/work/LMD-Beauty?id=67f63cbd76cbff918e936222',
 'https://10turtle.com/work/Pete-Evans?id=67f7a4fac2bb76a9dba597ce',
 'https://10turtle.com/work/Hungry-Tables?id=67f4b673f01b5aa64a300c5f',
 'https://10turtle.com/work/Cinci-420?id=67f5fa05d93fbbe04f69b719',
 'https://10turtle.com/work/AI-Assistant?id=67f4b185f01b5aa64a300b60',
 'https://10turtle.com/work/Smart-Cleanup?id=67f4b04ef01b5aa64a300a24',
 'https://10turtle.com/work/Digitaltrvst?id=67f60beaa3d1368640bc02cf',
 'https://10turtle.com/work/Prerra?id=67f50e7fb5adea113fa208d8',
 'https://10turtle.com/work/Halo-Notebook?id=67f51106b5adea113fa209c8',
 'https://10turtle.com/work/Fresh-Food?id=67f4be9781462886a9705cd8',
 'https://10turtle.com/work/Pure-Organic-Vitamins?id=67f51c3ab5adea113fa20ee3',
 'https://10turtle.com/work/-The-Estimators?id=67f50b59b5adea113fa207f8',
 'https://10turtle.com/work/Mina?id=67f4b96f81462886a9705b9a',
 'https://10turtle.com/work/Animart-AI?id=67f4c11d3fb33bf739e514b6',
 'https://10turtle.com/work/Golden-Touch-Design?id=67f384432a0c6d9d73acc7a0',
 'https://10turtle.com/work/-Free-to-Love?id=67f3af902a0c6d9d73accf8d',
 'https://10turtle.com/work/Pines-Edge?id=67f380112a0c6d9d73acc60e',
 'https://10turtle.com/work/StudyMate?id=67f39bb52a0c6d9d73accb48',
 'https://10turtle.com/work/The-Unda-Dog?id=67f374032a0c6d9d73acc176',
 'https://10turtle.com/work/Xora-AI?id=67f3c3022a0c6d9d73acd254',
 'https://10turtle.com/work/AI-Interior?id=67f3c7be2a0c6d9d73acd501',
 'https://10turtle.com/work/Smart-Alarm?id=67f3896b2a0c6d9d73acc8d2',
 'https://10turtle.com/work/DocsChat-AI?id=67f39f592a0c6d9d73acccf5',
 'https://10turtle.com/work/Gap-Life?id=67f3a7d32a0c6d9d73acce97',
 'https://10turtle.com/work/SmartChat-AI?id=67f3c0182a0c6d9d73acd0cc',
 'https://10turtle.com/work/Nav.it?id=67f3793e2a0c6d9d73acc39b',
 'https://10turtle.com/work/Vins-Dieux?id=67f0c9502a0c6d9d73acaf59',
 'https://10turtle.com/work/CryptaMiniatures?id=67f0e8632a0c6d9d73acb4dd'
 ]


work_docs = [Document(page_content=extract_title_from_work_url(url), metadata={"url": url}) for url in work_links]
work_texts = [doc.page_content for doc in work_docs]
work_embeddings = embedding_model.embed_documents(work_texts)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/predict")
def predict(data: ChatRequest):
    try:
        print("🔹 Step 1: Incoming question")
        question = data.question
        print("Question:", question)
 
        print("\n🔹 Step 2: Retrieving relevant documents")

        retriever = qa_chain.retriever

        docs = retriever.invoke(question)  # ← updated from get_relevant_documents

        print(f"Retrieved {len(docs)} documents.")

        for i, doc in enumerate(docs[:3]):

            print(f"  Doc {i+1}: {doc.page_content[:200]}...")
        
        print("\n🔹 Step 3: Generating prompt for LLM")

        context = "\n\n".join([d.page_content for d in docs])

        prompt_text = prompt_template.format(question=question, context=context)

        print("Prompt Preview:\n", prompt_text[:1000], "...\n")
        
        print("\n🔹 Step 4: Calling LLM with prompt")

        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")  # ← fix: don't try qa_chain.llm
        print(prompt_text)
        response_obj = llm.invoke(prompt_text)

        answer = getattr(response_obj, "content", str(response_obj))

 
 
        print("\n🔹 Step 5: Saving to conversation memory")
        memory.save_context({"question": question}, {"answer": answer})
        print("Memory updated. Chat history now has", len(memory.chat_memory.messages), "messages.")
 
        print("\n🔹 Step 6: Generating follow-up questions")
        followups = generate_followup_questions(question, answer)
        print("Follow-up questions:", followups)
 
        print("\n🔹 Step 7: Matching related work examples")
        query_embedding = embedding_model.embed_query(question + " " + answer)
        similarity_scores = [
            (cosine_similarity(query_embedding, emb), doc.metadata["url"])
            for emb, doc in zip(work_embeddings, work_docs)
        ]
        top_matches = sorted(similarity_scores, key=lambda x: x[0], reverse=True)
        related_work_matches = []
        for score, work_url in top_matches:
            if score > 0.5:
                related_work_matches.append(work_url)
            if len(related_work_matches) >= 5:
                break
        print("Top related works:", related_work_matches)
 
        print("\n🔹 Step 8: Extracting unique sources")
        unique_sources = set()
        unique_source_documents = []
        for doc in docs:
            source = doc.metadata.get("source")
            if source and source not in unique_sources:
                unique_sources.add(source)
                unique_source_documents.append({"source": source})
        print("Source documents:", unique_source_documents)
 
        print("\n🔹 Step 9: Formatting final answer")
        html_answer = convert_answer_to_html(answer)
 
        # print("\n✅ Completed QA process. Returning final response.")
        return {
            "question": question,
            "answer": html_answer,
            "follow_up_questions": followups,
            "source_documents": unique_source_documents,
            "related_works": [
                {"title": extract_title_from_work_url(url), "url": url}
                for url in related_work_matches
            ],
            "chat_history": memory.chat_memory.messages
        }
 
    except Exception as e:
        print("❌ Exception during QA process:", str(e))
        raise HTTPException(status_code=500, detail=f"Error occurred: {e}")
    
