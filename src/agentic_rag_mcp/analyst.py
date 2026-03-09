import logging
from typing import Dict, Any

from .provider import create_client_for
from .state import AgenticState

logger = logging.getLogger(__name__)

ANALYST_PROMPT = """You are a Senior System Analyst and the first line of defense in our engineering team.
Your ONLY job is to deeply understand the User's Query and translate it into a comprehensive "Intent Analysis & Investigation Plan" for the Manager.

YOU DO NOT HAVE TOOLS to search the code. You only have your brain and the Global Project Context.

YOUR OBJECTIVES:
1. **Empathy & Intent Deduction**: Why is the user asking this? Are they a frontend dev trying to integrate an API? Are they debugging an error?
2. **Contextual Expansion**: What "extra" information would be extremely helpful to them? (e.g. if they ask for a field name, they probably also want to know the validation rules, where to get the field value, and what HTTP errors might occur).
3. **Execution Plan**: Break down the investigation into 3-4 distinct logical steps (e.g. 1. Find the Controller, 2. Trace the DTO validation, 3. Check the Database interaction).

OUTPUT FORMAT (Plain Text):
Write a structured, highly professional analysis document. The Manager will read this and use it to direct the Workers.
"""

class AnalystAgent:
    def __init__(self):
        # 使用 reasoning 模型來確保最高品質的意圖推演
        self.client, self.comp_cfg = create_client_for("analyst")

    def act(self, state: AgenticState) -> Dict[str, Any]:
        logger.info("🧠 [Analyst] Deeply analyzing the user's intent and formulating a plan...")
        
        system_prompt = f"--- GLOBAL PROJECT CONTEXT ---\n{state.get('project_context', 'No context provided.')}\n\n====================\n{ANALYST_PROMPT}"
        user_content = f"USER QUERY: {state['query']}\n\nPlease provide your Intent Analysis & Investigation Plan."

        try:
            response = self.client.chat.completions.create(
                model=self.comp_cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3
            )
            analysis = response.choices[0].message.content
            logger.info(f"💡 [Analyst] Completed Analysis:\n{analysis[:500]}...")
            
            return {"intent_analysis": analysis}
        except Exception as e:
            logger.error(f"Analyst failed: {e}")
            return {"intent_analysis": "Analyst failed to produce a plan. Manager, please proceed with your best judgement."}
