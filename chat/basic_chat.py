from prompt_toolkit import PromptSession, print_formatted_text
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style


class ChatBot:
    def __init__(self):
        self.history = ChatMessageHistory()
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

    def get_response(self, query):
        self.history.add_user_message(query)
        response = self.model.invoke(self.history.messages).content
        self.history.add_ai_message(response)
        return response

    def handle_command(self, query):
        if query == "exit":
            exit("Bye!")
        elif query == "help":
            print_formatted_text(
                FormattedText(
                    [("class:info", "Available commands: exit, help, clear")]
                ),
                style=style,
            )
        elif query == "clear":
            self.history.clear()
            print_formatted_text(
                FormattedText([("class:info", "Chat history cleared")]), style=style
            )
        else:
            return False
        return True


style = Style.from_dict(
    {
        "ai": "#007bff bold",
        "user": "#28a745 bold",
        "info": "#777",
        "toolbar": "#eee",
    }
)

fragments = FormattedText(
    [
        ("class:ai", "AI:"),
    ]
)


def main():
    chatbot = ChatBot()
    session = PromptSession()
    while True:
        query = session.prompt(
            [("class:user", "User: ")],
            multiline=True,
            bottom_toolbar=[("class:toolbar", "Press Alt+Enter to submit")],
            style=style,
        )
        if query == "":
            continue
        if chatbot.handle_command(query):
            continue
        response = chatbot.get_response(query)
        print_formatted_text(fragments, response, style=style)


if __name__ == "__main__":
    main()
