from typing import Any
import tiktoken

class TokenTracker:
    """
    Tracks token usage for an AI agent.

    We track two things:

    1. Estimated context tokens
       - How many tokens are currently being sent to the LLM.

    2. Actual API usage
       - Input tokens reported by OpenAI.
       - Output tokens reported by OpenAI.
       - Total tokens reported by OpenAI.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model 

        # Get the tokenizer assosiate with the model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback if tiktoken does not recognize the model.
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # Current context information
        self.current_context_tokens = 0

        # API usage totals
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        # Request statistics
        self.total_requests = 0
        self.peak_context_tokens = 0


    # ------------------------Basic Token counting-------------------------
    def count_text(self , text : str) -> int:
        """
            Count token in a piece of text
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    # ---------------------------Message Token Counting----------------------------
    def count_message(self , messages : list[Any]) -> int:
        """
        Estimate the number of tokens contained in LangChain messages.
        This is an estimation of the context size.
        It is useful for comparing memory strategies because different
        memory systems will send different amounts of conversation
        history to the LLM.
        """
        total_tokens = 0 

        for message in messages:
            total_tokens += self.count_text(message.type)
            # Account for message content.
            content = message.content
            if isinstance(content, str):
                total_tokens += self.count_text(content)
            else:
                # Handle non-string content safely.
                total_tokens += self.count_text(str(content))

        return total_tokens

    # ----------------------------Context Tracking----------------------------
    def update_context(self , messages : list[Any]) -> int:
        """
            Calculate and store the current context size
        """
        self.current_context_tokens = self.count_message(messages)
        # Keep track of the largest context we seen
        self.peak_context_tokens = max(
            self.peak_context_tokens,
            self.current_context_tokens
        )
        return self.current_context_tokens

    # -------------------------------OpenAI Usage-----------------------------
    def record_api_usage(self, response: Any):
        """
        Record token usage returned by the LLM.

        LangChain responses normally expose usage information through
        response.usage_metadata.
        """
        usage = getattr(response,"usage_metadata",None)
        if not usage:
            return

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens",input_tokens + output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        self.total_requests += 1

    # -----------------------------Current Request report---------------------------------
    def get_current_report(self) -> dict:
        """
        Return statistics for the current context.
        """
        return {
            "context_tokens": self.current_context_tokens,
            "peak_context_tokens": self.peak_context_tokens,
        }

    # ---------------------------------Overall Report------------------------------------
    def get_total_report(self) -> dict:
        """
        Return cumulative API usage.
        """
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "peak_context_tokens": self.peak_context_tokens,
        }

    # ---------------------------------Pretty current report------------------------------------
    def print_current_report(self,messages: list[Any]):
        """
        Print token information for the current request.
        """
        context_tokens = self.update_context(messages)
        print("\n" + "-" * 50)
        print("TOKEN USAGE - CURRENT REQUEST")
        print("-" * 50)
        print(f"Messages in context : {len(messages)}")
        print(f"Context tokens      : {context_tokens}")
        print(f"Peak context tokens : {self.peak_context_tokens}")
        print("-" * 50)

    # ---------------------------------Pretty total report------------------------------------
    def print_total_report(self):
        """
        Print cumulative API usage.
        """
        report = self.get_total_report()

        print("\n" + "=" * 50)
        print("TOKEN USAGE - SESSION TOTAL")
        print("=" * 50)

        print(f"Requests: {report['total_requests']}")
        print(f"Input tokens: {report['total_input_tokens']}")
        print(f"Output tokens: {report['total_output_tokens']}")
        print(f"Total tokens: {report['total_tokens']}")
        print(f"Peak context tokens: {report['peak_context_tokens']}")
        print("=" * 50)

    # --------------------------------------Reset-------------------------------------------
    def reset(self):
        """
        Reset all token statistics.
        """

        self.current_context_tokens = 0

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

        self.total_requests = 0
        self.peak_context_tokens = 0