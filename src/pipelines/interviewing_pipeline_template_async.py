"""
TODO: Code template proposed by GPT
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import random
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel
from io import BytesIO
import json
import base64

# Import necessary async LLM and TTS/STT tools
from openai import AsyncOpenAI  # adjust as per actual import paths
from anthropic import AsyncAnthropic  # adjust as per actual import paths
from gtts import gTTS  # For simple TTS; modularize to support other tools

# Set up FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for tracking user session progress and configurations
user_states: Dict[str, Any] = {}


# TODO: Take class out to be a stand-alone module
class ReflectiveAgentAsync:
    async def generate_reflective_question(
        self, sub_concept_name: str, user_response: str
    ) -> str:
        prompts = [
            f"How might {sub_concept_name} impact real-world situations?",
            f"What alternative perspectives might exist around {sub_concept_name}?",
            f"Reflecting on your answer, would you change anything about {sub_concept_name}?",
        ]
        return random.choice(prompts)

    async def summarize_key_points(
        self,
        sub_concept: Dict[str, Any],
        conversation_memory: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        summary = f"We discussed '{sub_concept['name']}'. Key points included: {sub_concept.get('description', '')}."
        if conversation_memory:
            highlights = " ".join(
                f"- {ex['agent']} / {ex.get('user', '')}"
                for ex in conversation_memory[-3:]
            )
            summary += f" Recent highlights:\n{highlights}"
        return summary


# TODO: update this in facilitator_agent.py module
class FacilitatorAgentAsync:
    def __init__(
        self,
        user_id: str,
        data: Dict[str, Any],
        llm_provider: str,
        model_id: str,
        client: Any,
    ):
        self.user_id = user_id
        self.concept = data["concept"]
        self.sub_concepts = data["sub_concepts"]
        self.llm_provider = llm_provider
        self.model_id = model_id
        self.client = client
        self.current_sub_concept_index = 0
        self.conversation_memory = []
        self.reflective_agent = ReflectiveAgentAsync()

    async def start_conversation(self) -> str:
        return f"Let's discuss {self.concept}."

    async def discuss_next_sub_concept(self) -> Optional[str]:
        if self.current_sub_concept_index < len(self.sub_concepts):
            sub_concept = self.sub_concepts[self.current_sub_concept_index]
            self.current_sub_concept_index += 1
            return await self.generate_question(
                sub_concept["name"], sub_concept.get("description")
            )
        return None

    async def generate_question(
        self, sub_concept_name: str, sub_concept_description: Optional[str] = None
    ) -> str:
        prompt = f"Generate a question about '{sub_concept_name}'. Provide only the question."
        response = await self.client.completions.create(
            model=self.model_id, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


async def get_agent(user_id: str, data: Dict[str, Any]) -> FacilitatorAgentAsync:
    # Determine LLM client based on provider (defaulting to OpenAI)
    llm_provider = user_states[user_id].get("llm_provider", "openai")
    model_id = user_states[user_id].get("model_id", "gpt-4-turbo")
    if llm_provider == "openai":
        client = AsyncOpenAI(api_key="your-openai-key")
    elif llm_provider == "anthropic":
        client = AsyncAnthropic(api_key="your-anthropic-key")
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    return FacilitatorAgentAsync(
        user_id=user_id,
        data=data,
        llm_provider=llm_provider,
        model_id=model_id,
        client=client,
    )


async def modular_tts(text: str, provider: str = "gtts") -> str:
    if provider == "gtts":
        mp3_fp = BytesIO()
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return base64.b64encode(mp3_fp.read()).decode("utf-8")
    # Add more TTS providers here
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}")


class SkillModule:
    @staticmethod
    async def get_skills(user_id: str) -> List[str]:
        return ["Intermediate Machine Learning", "Basic Natural Language Processing"]


@app.get("/start/{user_id}")
async def start_conversation(user_id: str):
    try:
        data = {
            "concept": "Machine Learning",
            "sub_concepts": [
                {
                    "name": "Supervised Learning",
                    "description": "Learning from labeled data.",
                },
                {
                    "name": "Unsupervised Learning",
                    "description": "Finding patterns in data without labels.",
                },
            ],
        }
        user_states[user_id] = {
            "thought_index": 0,
            "sub_thought_index": 0,
            "llm_provider": "openai",
            "model_id": "gpt-4-turbo",
            "skills": await SkillModule.get_skills(user_id),
            "session_metadata": {"total_interactions": 0, "average_response_time": 0},
        }
        agent = await get_agent(user_id, data)
        start_message = await agent.start_conversation()
        return {"message": start_message}
    except Exception as e:
        logger.error(f"Error starting conversation: {e}")
        raise HTTPException(status_code=500, detail="Error starting conversation.")


@app.get("/get_question/{user_id}")
async def get_question(user_id: str):
    try:
        if user_id not in user_states:
            raise HTTPException(status_code=404, detail="User session not found.")
        data = {
            "concept": "Machine Learning",
            "sub_concepts": [
                {
                    "name": "Supervised Learning",
                    "description": "Learning from labeled data.",
                },
                {
                    "name": "Unsupervised Learning",
                    "description": "Finding patterns in data without labels.",
                },
            ],
        }
        agent = await get_agent(user_id, data)
        question = await agent.discuss_next_sub_concept()
        if question:
            user_states[user_id]["session_metadata"]["total_interactions"] += 1
            return {"question": question}
        else:
            summary = await agent.reflective_agent.summarize_key_points(
                data, agent.conversation_memory
            )
            return {"message": "Conversation complete", "summary": summary}
    except Exception as e:
        logger.error(f"Error retrieving question: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving question.")


@app.post("/synthesize_speech")
async def synthesize_speech(request: BaseModel):
    text, provider = request.text, request.provider
    try:
        audio_base64 = await modular_tts(text, provider)
        return {"audio": audio_base64, "content_type": "audio/mp3"}
    except Exception as e:
        logger.error(f"TTS processing error: {e}")
        raise HTTPException(status_code=500, detail="TTS processing failed.")
