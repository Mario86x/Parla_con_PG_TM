import asyncio
import os
import time
from dotenv import load_dotenv
import logging
from workflow import ChatWorkflow, UserMessageEvent, AssistantResponseEvent

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("debug_main")

async def run_debug_loop(api_key: str):
    wf = ChatWorkflow(api_key)
    logger.info("Debug workflow initialized")

    try:
        while True:
            msg = input("\n> You: ").strip()
            if not msg or msg.lower() in ("exit", "quit"):
                logger.info("User requested exit")
                break

            # 1) Create event and call generate_response step manually
            try:
                start = time.time()
                evt = UserMessageEvent(message=msg)
                logger.debug("Calling generate_response")
                resp = await wf.generate_response(None, evt)  # ctx not used in our implementation
                logger.debug("generate_response returned in %.3fs: %s", time.time() - start, type(resp))
            except Exception as e:
                logger.exception("Exception during generate_response")
                break

            # If returned StopEvent -> exit
            from workflows.workflow import StopEvent as _StopEvent  # safe check
            if resp is None or isinstance(resp, _StopEvent):
                logger.info("generate_response returned StopEvent / None -> exiting debug loop")
                break

            # 2) Display response (call step)
            try:
                # Use same invocation pattern as the workflow would
                start = time.time()
                ar_event = resp  # should be AssistantResponseEvent instance
                out = await wf.display_response(None, ar_event)
                logger.debug("display_response returned in %.3fs: %s", time.time() - start, type(out))
            except Exception as e:
                logger.exception("Exception during display_response")
                break

            # If display_response returned StopEvent -> exit
            if out is None or isinstance(out, _StopEvent):
                logger.info("display_response returned StopEvent / None -> exiting debug loop")
                break

            # If display_response returned a new UserMessageEvent, continue loop (it will ask input next cycle)
            # loop continues naturally
    except KeyboardInterrupt:
        logger.info("Debug loop interrupted by user")
    finally:
        logger.info("Debug loop ended")

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        raise SystemExit(1)
    asyncio.run(run_debug_loop(api_key))