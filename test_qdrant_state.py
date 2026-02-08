#!/usr/bin/env python3
"""
测试脚本：验证 QdrantStateStore 的所有功能
使用独立的测试 collection，不影响现有数据
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_qdrant_state_store():
    """测试 QdrantStateStore 的所有功能"""
    
    print("=" * 70)
    print("QdrantStateStore 功能测试")
    print("=" * 70)
    
    # 导入必要的模块
    try:
        from qdrant_client import QdrantClient
        from agentic_rag_mcp.indexer.qdrant_state_store import QdrantStateStore
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("\n请安装依赖: pip install qdrant-client")
        return False
    
    # 连接到 Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"\n📡 连接到 Qdrant: {qdrant_url}")
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        # 测试连接
        collections = client.get_collections()
        print(f"✓ 成功连接 (现有 {len(collections.collections)} 个 collections)")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False
    
    # 使用测试 collection
    test_collection = "test-qdrant-state"
    print(f"\n🧪 使用测试 collection: {test_collection}")
    print(f"   State collection 将是: {test_collection}-state")
    
    # 初始化 QdrantStateStore
    print("\n1️⃣ 初始化 QdrantStateStore...")
    try:
        state_store = QdrantStateStore(client=client, main_collection=test_collection)
        print("   ✓ QdrantStateStore 初始化成功")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 1: 保存和加载全局状态
    print("\n2️⃣ 测试全局状态管理...")
    try:
        # 保存
        state_store.save_global_state(
            embedding_model="test-embedding-model-v1",
            sparse_mode="bm25"
        )
        print("   ✓ 全局状态已保存")
        
        # 加载
        global_state = state_store.load_global_state()
        assert global_state is not None, "全局状态不应为 None"
        assert global_state["embedding_model"] == "test-embedding-model-v1"
        assert global_state["sparse_mode"] == "bm25"
        print(f"   ✓ 全局状态已加载: {global_state['embedding_model']}, {global_state['sparse_mode']}")
        
    except Exception as e:
        print(f"   ❌ 全局状态测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 2: 保存和获取文件状态
    print("\n3️⃣ 测试文件状态管理...")
    test_files = [
        ("src/main.py", "hash123abc", 5),
        ("src/utils.py", "hash456def", 3),
        ("tests/test_main.py", "hash789ghi", 2),
    ]
    
    try:
        # 保存多个文件状态
        for file_path, file_hash, chunks in test_files:
            state_store.save_file_state(
                file_path=file_path,
                file_hash=file_hash,
                chunks=chunks
            )
        print(f"   ✓ 已保存 {len(test_files)} 个文件状态")
        
        # 获取文件状态
        for file_path, expected_hash, expected_chunks in test_files:
            file_state = state_store.get_file_state(file_path)
            assert file_state is not None
            assert file_state["file_path"] == file_path
            assert file_state["hash"] == expected_hash
            assert file_state["chunks"] == expected_chunks
        print(f"   ✓ 所有文件状态验证成功")
        
    except Exception as e:
        print(f"   ❌ 文件状态测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 3: 统计文件数量
    print("\n4️⃣ 测试文件计数...")
    try:
        count = state_store.count_indexed_files()
        assert count == len(test_files), f"期望 {len(test_files)} 个文件，实际 {count} 个"
        print(f"   ✓ 文件计数正确: {count} 个文件")
    except Exception as e:
        print(f"   ❌ 文件计数测试失败: {e}")
        return False
    
    # 测试 4: 更新最后索引时间
    print("\n5️⃣ 测试更新最后索引时间...")
    try:
        state_store.update_last_index_time()
        updated_state = state_store.load_global_state()
        assert "last_index_time" in updated_state
        print(f"   ✓ 最后索引时间已更新: {updated_state['last_index_time']}")
    except Exception as e:
        print(f"   ❌ 更新时间测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 5: 获取所有文件
    print("\n6️⃣ 测试获取所有已索引文件...")
    try:
        all_files = state_store.get_all_indexed_files()
        assert len(all_files) == len(test_files)
        print(f"   ✓ 获取到 {len(all_files)} 个文件:")
        for f in all_files:
            print(f"      - {f['file_path']}: {f['chunks']} chunks, hash={f['hash'][:8]}...")
    except Exception as e:
        print(f"   ❌ 获取所有文件测试失败: {e}")
        return False
    
    # 测试 6: 删除文件状态
    print("\n7️⃣ 测试删除文件状态...")
    try:
        delete_file = test_files[0][0]  # 删除第一个文件
        state_store.delete_file_state(delete_file)
        print(f"   ✓ 已删除文件状态: {delete_file}")
        
        # 验证已删除
        deleted_state = state_store.get_file_state(delete_file)
        assert deleted_state is None, "文件状态应该被删除"
        print(f"   ✓ 验证删除成功")
        
        # 验证计数
        new_count = state_store.count_indexed_files()
        assert new_count == len(test_files) - 1
        print(f"   ✓ 文件计数更新: {new_count} 个文件")
        
    except Exception as e:
        print(f"   ❌ 删除文件测试失败: {e}")
        return False
    
    # 清理：删除测试 collection
    print("\n8️⃣ 清理测试数据...")
    try:
        state_store.clear_all_states()
        print(f"   ✓ 测试 collection 已清理")
    except Exception as e:
        print(f"   ⚠️  清理警告: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    print("\n💡 QdrantStateStore 功能正常，可以安全使用")
    print(f"   测试 collection: {test_collection}-state 已清理")
    
    return True

if __name__ == "__main__":
    try:
        success = test_qdrant_state_store()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
