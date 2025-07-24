from llama_index.core.workflow import Workflow, step, Context, Event, StartEvent, StopEvent
from llm import init_llm
from templates import SYSTEM_PROMPT

class UserMessageEvent(Event):
    message: str

class AssistantResponseEvent(Event):
    response: str

class ChatWorkflow(Workflow):
    def __init__(self, api_key: str):
        super().__init__()
        self.llm = init_llm(api_key)

    async def _update_running_story(self, ctx: Context, new_content: str):
        running_story = await ctx.get("running_story", "")
        running_story += f"\n\n{new_content}"
        await ctx.set("running_story", running_story)

    @step
    async def get_user_message(self, ctx: Context, ev: StartEvent | AssistantResponseEvent) -> UserMessageEvent:
        if isinstance(ev, StartEvent):
            print("Starting the chat...")
        else:
            print("\nPrevious response:")
            print(ev.response)

        user_message = input("\nYour message: ")
        await self._update_running_story(ctx, f"\nUser: {user_message}")
        return UserMessageEvent(message=user_message)

    @step
    async def generate_response(self, ctx: Context, ev: UserMessageEvent) -> AssistantResponseEvent | StopEvent:
        running_story = await ctx.get("running_story", "")

        prompt = f"{SYSTEM_PROMPT.template}\n\n{running_story}\n\nYou: {ev.message}\nAssistant:"

        print(f"\nGenerating response for: {ev.message}\n--------------\n")

        #counting tokens in prompt:
        token_count = self.llm.count_tokens(prompt)
        print(f"Token count for the prompt: {token_count}")

        try:
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            await self._update_running_story(ctx, f"\nAssistant: {response_text}")
            return AssistantResponseEvent(response=response_text)
        except Exception as e:
            print(f"Error generating response: {e}")
            return StopEvent()

    # @step
    # async def should_continue(self, ctx: Context, ev: AssistantResponseEvent) -> StopEvent | AssistantResponseEvent:
    #     continue_chat = input("\nContinue chatting? (yes/no): ").lower()
    #     if continue_chat == "yes":
    #         return ev
    #     else:
    #         print("Ending the chat.")
    #         return StopEvent()