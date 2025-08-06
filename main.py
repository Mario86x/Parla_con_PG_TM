import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from workflow import ChatWorkflow


# Initialize Rich console for better output
console = Console()



async def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[bold red]Error: GOOGLE_API_KEY not found in environment variables[/bold red]")
        return

    try:
        # Initialize workflow
        workflow = ChatWorkflow(api_key)
        console.print("[bold green]Starting your chat...[/bold green]\n")

        # Run the chat workflow
        await workflow.run()

        console.print("\n[bold green]Chat completed![/bold green]")

    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
        return

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Chat ended by user[/bold yellow]")