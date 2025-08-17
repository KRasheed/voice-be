import os, asyncio
from dotenv import load_dotenv
from livekit.agents import cli, WorkerOptions, JobContext
from livekit import rtc
# from openai import AsyncOpenAI
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from livekit.agents import cli, WorkerOptions, JobContext
from livekit.agents import Agent, AgentSession, RoomInputOptions, RoomOutputOptions, RunContext
from livekit.plugins import openai
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, silero, noise_cancellation
import random
load_dotenv()

client = OpenAI()

# --- Config ---
LK_URL         = os.getenv("LIVEKIT_URL")
LK_API_KEY     = os.getenv("LIVEKIT_API_KEY")
LK_API_SECRET  = os.getenv("LIVEKIT_API_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PINECONE_API_KEY   = 'pcsk_5tj3rr_KWyW2gpj4mkUWckLb64HsZnLDwBPKMxpoEsFyzoSdZfUdbGLb5dgD2cewbZq4vb'
PINECONE_NAMESPACE = "default_namespace"
PINECONE_INDEX     = "instruments"
EMBED_MODEL        = "text-embedding-3-small"
TOP_K              = 2  # integer

# ---- Clients ----
# llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
pc  = Pinecone(api_key=PINECONE_API_KEY)


# init embedding model
embedding = OpenAIEmbeddings(model=EMBED_MODEL)

from pinecone import ServerlessSpec

index_name = PINECONE_INDEX  # "instruments"

# Ensure index exists
existing = [i["name"] for i in pc.list_indexes().indexes]
print(existing)
if PINECONE_INDEX not in existing:
    raise RuntimeError(f"Index '{PINECONE_INDEX}' not found in Pinecone. Found: {existing}")

index = pc.Index(PINECONE_INDEX)
vectorstore = PineconeVectorStore(index=index, embedding=embedding, namespace=PINECONE_NAMESPACE)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


async def retrieve_from_pinecone(query: str) -> str:
    """Fetch relevant docs from Pinecone and return as a context string."""

    docs = retriever.invoke(query)
    
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

# ----------------------
# LLM setup (GPT-4o-nano)
# ----------------------
# llm = LKOpenAI(
#     model="gpt-4o-nano",   # Fast, streaming-friendly model
#     api_key=OPENAI_API_KEY,
#     temperature=0.2
# )

# ----------------------
# RAG Agent definition
# # ----------------------
class RagAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""
        You are Guitar center AI assistant, a helpful IT assistant who communicates through voice.
                     
        Voice Communication Guidelines:
        - Speak naturally as you would in a conversation, not like you're reading text
        - Keep responses brief and focused (20-40 words)
        - Use simple sentence structures that are easy to follow when heard
        - Avoid long lists, complex numbers, or detailed technical terms unless necessary
        - Use natural transitions and conversational markers
        - Nevery makethings up.
        
        Remember: You're having a conversation, not reading a document.""",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4.1-mini"), 
            tts=openai.TTS(instructions="You are Guitar center AI assistant, a helpful assistant. Speak naturally as if having a casual conversation, not reading from a document.",
                       model="gpt-4o-mini-tts",
                       voice="coral",
                       response_format="pcm",)
            )
        

    
    @function_tool
    async def lookup_info(self, context: RunContext, query: str):
        """
        ALWAYS use this function to look up information using RAG when the user asks a question
        about a topic that might be in our knowledge base. Nevery makethings up.
        
        Args:
            query: The question or topic to look up
        """
       
    
        # Tell the user we're looking things up
        thinking_messages = [
            "Let me look that up...",
            "One moment while I check...",
            "I'll find that information for you...",
            "Just a second while I search...",
            "Looking into that now..."
        ]
        await self.session.say(random.choice(thinking_messages))
        
        try:

            docs = retriever.invoke(query)
            context = docs[0].page_content
            
            if not docs:
                return None, "I couldn't find any relevant information about that."
                
            # Get the most relevant paragraph
            # paragraph = self._paragraphs_by_uuid.get(results[0].userdata, "")
            
            if not context:
                return None, "I couldn't find any relevant information about that."
            
            # Generate response with context
            context_prompt = f"""
            Question: {query}
            
            Relevant information:
            {context}
            
            Using the relevant information above, please provide a helpful response to the question.
            Keep your response concise, short, to the point, and directly answer the question. Nevery makethings up.
            """
            
            response = client.responses.create(
                    model="gpt-4.1-nano",
                    input=context_prompt
                )
            print("-------------",response.output_text,"-------------------")

            return None, response.output_text
        except Exception as e:
            # logger.error(f"Error during RAG lookup: {e}")
            return None, "I encountered an error while trying to look that up."
        

        
    # async def on_text(self, text: str):
    #     # 1) Retrieve context from Pinecone
    #     print(text)
    #     ctx_text = await retrieve_from_pinecone(text)
    #     print(f'--------------------Context {ctx_text}-----------------')
    #     # 2) Build RAG prompt
    #     prompt = f"Question: {text}\n\nContext:\n{ctx_text}"

    #     print("prompt: ", prompt)
    #     # 3) Stream answer token-by-token
    #     response = client.responses.create(
    #                 model="gpt-4.1-nano",
    #                 input=prompt
    #             )
    
    #     return None, response.output_text


# ----------------------
# Entrypoint
# ----------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(vad=silero.VAD.load(),)
    agent = RagAgent()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
        # room_input_options=RoomInputOptions(
        #     text_enabled=True,   # Type in console
        #     audio_enabled=True   # Speak via mic
        # ),
        # room_output_options=RoomOutputOptions(
        #     transcription_enabled=False,
        #     audio_enabled=False  # Change to True if you want TTS output
        # ),
    )
    await session.generate_reply(
        instructions="Greet the user by introduing yourself and offer your assistance."
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))