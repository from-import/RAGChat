# RAGChat
RAG问答系统

2025.1.3 运行成功
![608310d918d1256e3a3e541a1e809571](https://github.com/user-attachments/assets/3dfdc6b9-de57-41b8-92fe-720aca38a747)
![7c792f60f89c68ee6f9b2b60efcfb58f](https://github.com/user-attachments/assets/a8f48f01-456a-40d9-9fb3-1f57cc93121e)
![eab2ea38b44bab4290648d26ba19c6a5](https://github.com/user-attachments/assets/264a8641-7e73-429b-aa0e-d62e5235d839)
![030f2ec356f1094bd70cd570b04bfdcb](https://github.com/user-attachments/assets/a7350083-0ec2-4a4b-aff3-6b8008f494b9)

运行方式：
0. (cmd)激活虚拟环境
venv\Scripts\activate  

1. 构建知识图谱  
python index.py --root ./ragtest

2. 执行查询
python query.py --root ./ragtest --method local "What are the top themes in this story?"  
python query.py --root ./ragtest --method local "Who is Bob Cratchit"

