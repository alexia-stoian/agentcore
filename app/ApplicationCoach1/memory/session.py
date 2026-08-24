import os
import uuid
from typing import Optional

from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

MEMORY_ID = os.getenv("MEMORY_APPLICATIONCOACH1MEMORY_ID")
REGION = os.getenv("AWS_REGION")

def get_memory_session_manager(session_id: Optional[str], actor_id: str) -> Optional[AgentCoreMemorySessionManager]:
    if not MEMORY_ID:
        return None

    # AgentCoreMemoryConfig rejects None; OAuth/CUSTOM_JWT callers can reach us
    # without a runtime session header, so synthesize one when absent.
    session_id = session_id or uuid.uuid4().hex

    # Retrieve the user's long-term memory (semantic facts + preferences, keyed by actor across
    # sessions) so agents recall what they learned in earlier conversations, plus the current
    # session's episodes for in-conversation recall. The authoritative user_profile is still
    # injected on top of this every turn.
    retrieval_config = {
        f"/users/{actor_id}/facts": RetrievalConfig(top_k=5, relevance_score=0.4),
        f"/users/{actor_id}/preferences": RetrievalConfig(top_k=5, relevance_score=0.4),
        f"/episodes/{actor_id}/{session_id}": RetrievalConfig(top_k=3, relevance_score=0.5),
    }

    return AgentCoreMemorySessionManager(
        AgentCoreMemoryConfig(
            memory_id=MEMORY_ID,
            session_id=session_id,
            actor_id=actor_id,
            retrieval_config=retrieval_config,
        ),
        REGION
    )

