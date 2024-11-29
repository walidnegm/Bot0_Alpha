"""" TBA """

import logging
import random
from typing import Dict, Any, Optional, List, Union
import asyncio


# TODO: still working on
class ReflectiveAgentAsync:
    def __init__(self):
        self = self

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
