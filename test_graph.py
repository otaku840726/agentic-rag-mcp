import os
from dotenv import load_dotenv

# Ensure we have environment variables
load_dotenv()

from src.agentic_rag_mcp.agentic_search import AgenticSearch

def test_agentic_search():
    print("=" * 50)
    print("🚀 Starting LangGraph Agentic Search Test")
    print("=" * 50)

    query = "member deposit create api, merchantCode 應該填什麼？"
    print(f"User Query: {query}\n")

    # Initialize the new state graph agent
    agent = AgenticSearch()

    # Execute the search graph
    result = agent.search(query)

    if result.success:
        print("\n✅ Search Completed Successfully!")
        print("=" * 50)

        resp = result.response
        print(f"🎯 Answer:\n{resp.answer}\n")

        if resp.flow:
            print("🔄 Flow:")
            for f in resp.flow:
                print(f"  {f.step}. {f.description}")

        print(f"\n📈 Iterations: {resp.iterations}")
        print(f"🗂️  Total Evidence Found: {resp.total_evidence_found}")

        print("\n🛠️ Graph Internal Debug Info:")
        debug_info = result.debug_info
        print(f"Search ID: {debug_info.get('search_id')}")

        perf = debug_info.get("perf_stats", {})
        print(f"Time Elapsed: {perf.get('elapsed_ms')} ms")

    else:
        print("\n❌ Search Failed!")
        print(f"Error: {result.error}")
        print(f"Traceback:\n{result.debug_info.get('traceback', '')}")

if __name__ == "__main__":
    test_agentic_search()
