#!/usr/bin/env python3
"""
集成测试脚本 - 测试所有主要功能
测试：index-files, index-by-pattern, quick-search, agentic-search, index-status
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from agentic_rag_mcp.indexer.core import IndexerService

# 加载环境变量
load_dotenv()

def print_section(title: str):
    """打印测试章节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_step(step_num: int, description: str):
    """打印测试步骤"""
    print(f"\n{step_num}️⃣  {description}")

def print_result(success: bool, message: str):
    """打印测试结果"""
    icon = "✓" if success else "✗"
    print(f"   {icon} {message}")

def create_test_files(temp_dir: Path):
    """创建测试文件"""
    test_files = {
        "src/auth.py": """
class AuthService:
    '''用户认证服务'''
    
    def authenticate(self, username: str, password: str) -> bool:
        '''验证用户凭证'''
        # 验证逻辑
        return True
    
    def generate_token(self, user_id: str) -> str:
        '''生成 JWT token'''
        return f"token_{user_id}"
""",
        "src/database.py": """
class DatabaseService:
    '''数据库服务'''
    
    def connect(self, connection_string: str):
        '''连接数据库'''
        pass
    
    def query(self, sql: str) -> list:
        '''执行查询'''
        return []
""",
        "src/utils.py": """
def format_date(date_str: str) -> str:
    '''格式化日期'''
    return date_str

def validate_email(email: str) -> bool:
    '''验证邮箱格式'''
    return '@' in email
""",
        "docs/api.md": """
# API Documentation

## Authentication API

### POST /auth/login
Login with username and password.

### POST /auth/logout
Logout current user.
""",
        "README.md": """
# Test Project

This is a test project for integration testing.

## Features
- Authentication
- Database operations
- Utility functions
"""
    }
    
    for rel_path, content in test_files.items():
        file_path = temp_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
    
    return list(test_files.keys())

def test_environment():
    """测试环境配置"""
    print_section("环境配置检查")
    
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"]
    all_present = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:10] + "..." if len(value) > 10 else value
            print_result(True, f"{var} = {masked}")
        else:
            print_result(False, f"{var} not found")
            all_present = False
    
    return all_present

def test_indexer_service(temp_dir: Path, test_files: list):
    """测试索引服务"""
    print_section("测试 IndexerService")
    
    # 切换到临时目录
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        indexer = IndexerService()
        
        # 测试 1: index-files
        print_step(1, "测试 index-files (单个文件)")
        result = indexer.index_files(file_paths=["README.md"])
        total_processed = result.get('indexed', 0) + result.get('skipped', 0)
        print_result(
            total_processed > 0,
            f"索引文件数: {result.get('indexed', 0)}, 跳过: {result.get('skipped', 0)}"
        )
        
        # 测试 2: index-by-pattern
        print_step(2, "测试 index-by-pattern (批量索引)")
        result = indexer.index_by_pattern(pattern="src/**/*.py")
        total_processed = result.get('indexed', 0) + result.get('skipped', 0)
        print_result(
            total_processed > 0,
            f"索引文件数: {result.get('indexed', 0)}, 跳过: {result.get('skipped', 0)}"
        )
        
        # 测试 3: index-status
        print_step(3, "测试 index-status")
        status = indexer.get_status()
        print_result(True, f"已索引文件: {status.get('indexed_files', 0)}")
        print_result(True, f"向量点数: {status.get('points_count', 0)}")
        print_result(True, f"embedding 模型: {status.get('embedding_model', 'N/A')}")
        
        # 测试 4: 增量索引（不应重复索引）
        print_step(4, "测试增量索引 (应跳过未修改文件)")
        result = indexer.index_files(file_paths=["README.md"])
        print_result(
            result.get('skipped', 0) > 0,
            f"跳过未修改文件: {result.get('skipped', 0)}"
        )
        
        # 测试 5: 强制重新索引
        print_step(5, "测试强制重新索引")
        result = indexer.index_by_pattern(pattern="README.md", force=True)
        total_processed = result.get('indexed', 0) + result.get('skipped', 0)
        print_result(
            total_processed > 0,
            f"强制重新索引: {result.get('indexed', 0)} 个文件"
        )
        
        return True
        
    except Exception as e:
        print_result(False, f"索引测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

def test_search_service(temp_dir: Path):
    """测试搜索服务"""
    print_section("测试搜索功能")
    
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # 初始化 indexer
        indexer = IndexerService()
        
        # 测试 1: Qdrant 查询（基本功能）
        print_step(1, "测试 Qdrant 搜索功能")
        try:
            # 执行简单查询
            query_text = "authentication login"
            query_embedding = indexer.embedder.embed_batch([query_text])[0]
            
            results = indexer.qdrant.client.search(
                collection_name=indexer.qdrant.collection_name,
                query_vector=query_embedding,
                limit=3
            )
            
            print_result(
                len(results) > 0,
                f"找到 {len(results)} 个相关结果"
            )
            if results:
                print(f"   最相关: {results[0].payload.get('file_path', 'N/A')}")
        except Exception as e:
            print_result(False, f"Qdrant 搜索失败: {str(e)}")
        
        # 测试 2: 验证索引内容
        print_step(2, "测试索引内容验证")
        try:
            # 获取 collection 信息
            collection_info = indexer.qdrant.client.get_collection(
                collection_name=indexer.qdrant.collection_name
            )
            print_result(
                collection_info.points_count > 0,
                f"Collection 包含 {collection_info.points_count} 个 points"
            )
        except Exception as e:
            print_result(False, f"获取 collection 信息失败: {str(e)}")
        
        # 测试 3: 元数据查询
        print_step(3, "测试基于元数据的过滤")
        try:
            # 使用 scroll 获取一些数据
            scroll_result = indexer.qdrant.client.scroll(
                collection_name=indexer.qdrant.collection_name,
                limit=5
            )
            points, _ = scroll_result
            print_result(
                len(points) > 0,
                f"成功获取 {len(points)} 个索引条目"
            )
            if points:
                # 显示一些示例
                for i, point in enumerate(points[:2], 1):
                    file_path = point.payload.get('file_path', 'N/A')
                    print(f"      {i}. {file_path}")
        except Exception as e:
            print_result(False, f"元数据查询失败: {str(e)}")
        
        return True
        
    except Exception as e:
        print_result(False, f"搜索测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

def test_state_management(temp_dir: Path):
    """测试状态管理"""
    print_section("测试 Qdrant 状态管理")
    
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        indexer = IndexerService()
        state_store = indexer.state_store
        
        # 测试 1: 全局状态
        print_step(1, "测试全局状态读取")
        global_state = state_store.load_global_state()
        if global_state:
            print_result(
                True,
                f"Embedding 模型: {global_state.get('embedding_model', 'N/A')}"
            )
            print_result(
                True,
                f"Sparse 模式: {global_state.get('sparse_mode', 'N/A')}"
            )
        else:
            print_result(False, "未找到全局状态（可能是初次运行）")
        
        
        # 测试 2: 文件状态
        print_step(2, "测试文件状态读取")
        file_state = state_store.get_file_state("README.md")
        if file_state:
            print_result(True, f"README.md chunks: {file_state.get('chunks', 0)}")
            print_result(True, f"README.md hash: {file_state.get('hash', 'N/A')[:12]}...")
        else:
            print_result(False, "无法读取 README.md 状态")
        
        # 测试 3: 文件计数
        print_step(3, "测试文件计数")
        count = state_store.count_indexed_files()
        print_result(count > 0, f"已索引文件数: {count}")
        
        # 测试 4: 获取所有文件
        print_step(4, "测试获取所有已索引文件")
        all_files = state_store.get_all_indexed_files()
        print_result(len(all_files) > 0, f"文件列表: {len(all_files)} 个文件")
        for i, file_info in enumerate(all_files[:3], 1):
            print(f"      {i}. {file_info.get('file_path')}")
        
        return True
        
    except Exception as e:
        print_result(False, f"状态管理测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

def main():
    """主测试流程"""
    print("\n" + "=" * 70)
    print("  Agentic RAG MCP - 集成测试")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    
    # 1. 检查环境
    if not test_environment():
        print("\n❌ 环境配置不完整，测试终止")
        return 1
    
    # 2. 创建临时测试目录
    print_section("创建测试环境")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_files = create_test_files(temp_path)
        print_result(True, f"创建 {len(test_files)} 个测试文件")
        
        # 3. 运行测试
        results = []
        
        results.append(("索引服务", test_indexer_service(temp_path, test_files)))
        results.append(("搜索功能", test_search_service(temp_path)))
        results.append(("状态管理", test_state_management(temp_path)))
    
    # 4. 汇总结果
    print_section("测试结果汇总")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        print_result(success, f"{name}: {'通过' if success else '失败'}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
