from app.chatbot.chatbot_state import ChatbotState

class ChatbotEngine:
    def __init__(self):
        self.state = ChatbotState()

    def handle_message(self, message: str) -> str:
        message = message.strip()

        # If we don't know the user's name yet
        if self.state.user_name is None:
            if message == "":
                return "Hi! 👋 What’s your name?"
            else:
                self.state.user_name = message
                return f"Nice to meet you, {self.state.user_name}! 😊\nYou can ask me about F1 strategy whenever you’re ready."

        # After name is known
        return f"{self.state.user_name}, I’m still warming up 🚦\nTry asking me something soon!"
