@echo off
启动后端服务
start cmd /k "python app.py --port 5000"

启动前端应用
start cmd /k "cd frontend && set PORT=3000 && npm start"

启动二分类应用
start cmd /k "cd SVM_binary && streamlit run app.py --server.port 8501"

启动多分类应用
start cmd /k "cd SVM_multi && streamlit run app.py --server.port 8502"