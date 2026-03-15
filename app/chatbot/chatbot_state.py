from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatbotState:
    user_name: Optional[str] = None
