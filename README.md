uvicorn src.main:app --host 0.0.0.0 --port 8090 --reload

# if __name__ == "__main__" :
#     uvicorn.run(app, host="0.0.0.0", port=8090, reload=True)

pip freeze > requirements.txt
pip install -r 파일이름.txt



용어 정리
CGI(Common Gateway Interface)
was에 동적인 요청이 들어왔을 때 공통 규약(인터페이스)을 제공

WSGI(Web Server Gateway Interface)
CGI의 단점(요청마다 새로운 프로세스 생성 등)을 보완
callable object 등으로 요청 처리

ASGI(Asynchronous Server Gateway Interface)
WSGI의 단점(비동기 처리에 단점)을 보완
Uvicorn이 ASGI에서 활용

변수로서 경로를 넣는 방법
/files/{file_path:path}
