copilot中的claude模型和claude code模型的区别：
**VS Code Copilot 中使用 Claude 模型**                                                                      
  - **系统提示词和工具调用逻辑由 GitHub 控制**，不是 Anthropic 设计的
  - 上下文窗口管理、代码索引方式都是 Copilot 自己的架构
  - 所以同样的 Claude 模型，表现可能不如直接使用 Anthropic 的产品
  "降智"这个说法不太准确——**模型本身没有变**，但由于中间层的 prompt 设计和上下文处理不同，实际效果可能打折扣。
  **Claude** **Code** **VS** **Code** **扩展**
  **不会有这个问题。** Claude Code 扩展本质上就是把 Claude Code CLI 集成到了 VS Code 的界面里：
  - 直接调用 **Anthropic** **的** **API**
  - 使用的是 **Anthropic** **自己设计的系统提示词和工具链**
  - 和你在终端里运行 claude 命令是**完全一样的体验**
  - 没有第三方中间层

![[Pasted image 20260306122209.png]]