#!/usr/bin/env python3
"""
Quick verification script for Qdrant state management integration.
Tests basic read/write operations with the new QdrantStateStore.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentic_rag_mcp.indexer.qdrant_state_store import QdrantStateStore
from qdrant_client import QdrantClient

def main():
    print("=" * 60)
    print("Qdrant State Management Verification")
    print("=" * 60)
    
    # Connect to Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "optimuspay-hybrid")
    
    print(f"\n1. Connecting to Qdrant: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    print(f"2. Initializing QdrantStateStore for collection: {collection_name}")
    state_store = QdrantStateStore(client=client, main_collection=collection_name)
    
    # Test 1: Global state
    print("\n--- Test 1: Global State Management ---")
    print("Saving global state...")
    state_store.save_global_state(
        embedding_model="test-model-v1",
        sparse_mode="bm25"
    )
    
    print("Loading global state...")
    global_state = state_store.load_global_state()
    print(f"✓ Global state loaded: {global_state}")
    
    assert global_state["embedding_model"] == "test-model-v1"
    assert global_state["sparse_mode"] == "bm25"
    print("✓ Global state test passed")
    
    # Test 2: File state
    print("\n--- Test 2: File State Management ---")
    test_file = "test/example.py"
    test_hash = "abc123def456"
    test_chunks = 5
    
    print(f"Saving file state for: {test_file}")
    state_store.save_file_state(
        file_path=test_file,
        file_hash=test_hash,
        chunks=test_chunks
    )
    
    print(f"Loading file state for: {test_file}")
    file_state = state_store.get_file_state(test_file)
    print(f"✓ File state loaded: {file_state}")
    
    assert file_state["file_path"] == test_file
    assert file_state["hash"] == test_hash
    assert file_state["chunks"] == test_chunks
    print("✓ File state test passed")
    
    # Test 3: Count files
    print("\n--- Test 3: File Counting ---")
    count = state_store.count_indexed_files()
    print(f"✓ Indexed files count: {count}")
    assert count >= 1  # At least our test file
    
    # Test 4: Update last index time
    print("\n--- Test 4: Update Last Index Time ---")
    print("Updating last index time...")
    state_store.update_last_index_time()
    
    updated_global = state_store.load_global_state()
    assert "last_index_time" in updated_global
    print(f"✓ Last index time updated: {updated_global['last_index_time']}")
    
    # Test 5: Get all indexed files
    print("\n--- Test 5: Get All Indexed Files ---")
    all_files = state_store.get_all_indexed_files()
    print(f"✓ Total indexed files: {len(all_files)}")
    for f in all_files[:3]:  # Show first 3
        print(f"  - {f['file_path']}: {f['chunks']} chunks")
    
    # Cleanup test file
    print("\n--- Cleanup ---")
    print(f"Deleting test file state: {test_file}")
    state_store.delete_file_state(test_file)
    
    deleted_state = state_store.get_file_state(test_file)
    assert deleted_state is None
    print("✓ Test file state deleted successfully")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
