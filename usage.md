这些测试文件只需要启动 server 和 router 就可以运行，不需要运行 send_request_and_track.py。每个测试文件有不同的用途：

  1. test_simple_timestamp.py - 最简单的测试

  # 只需要 server 和 router 已启动
  python test_simple_timestamp.py

  # 或指定服务器地址
  python test_simple_timestamp.py http://localhost:40009 http://localhost:40005 http://localhost:40006
  用途：快速检查服务器响应中是否包含时间戳字段。

  2. test_timestamp_tracking.py - 全面测试

  # 使用默认地址
  python test_timestamp_tracking.py

  # 或指定地址
  python test_timestamp_tracking.py --router-url http://localhost:40009 --server-urls http://localhost:40005
  http://localhost:40006
  用途：全面测试各种端点和响应格式，包括流式响应。

  3. check_server_version.py - 版本验证

  python check_server_version.py
  用途：检查服务器是否使用了包含调试版本标记的代码。

  4. test_direct_api.py - API 端点测试

  python test_direct_api.py
  用途：测试不同的 API 端点和请求格式，查找时间相关字段。

  测试流程

  1. 确保服务已启动
  # 在远程服务器上
  # 确认 server 在 40005, 40006 端口运行
  # 确认 router 在 40009 端口运行
  2. 运行简单测试（推荐先运行这个）
  python test_simple_timestamp.py
  3. 如果没有看到时间戳，需要：
    - 确认远程服务器的 tokenizer_manager.py 已包含时间戳修改
    - 重启服务器
    - 运行版本检查：python check_server_version.py
  4. 运行全面测试
  python test_timestamp_tracking.py

  这些测试工具是独立的，可以随时运行，不依赖于 send_request_and_track.py。它们直接向服务器发送 HTTP 请求并分析响应。