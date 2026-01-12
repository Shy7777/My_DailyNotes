### 1. Q：在git提交至GitHub时推送出现：Failed to connect to github.com port 443 after 21112 ms: Couldn't connect to server  /  Recv failure: Connection was reset

### A：修改代理端口，在cmd中输入`git config --global http.proxy http://127.0.0.1:4780,其中7890需要在系统的网络和Internet中的高级网络属性->硬件链接属性最下方找到端口