from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import time

app = FastAPI()

# In-memory storage for OCR results
result = None

def send_data_to_next(json_data):
    # Next.js 서버의 엔드포인트로 JSON 데이터를 POST 요청으로 전송
    nextjs_url = "http://localhost:3000/api/receive"
    response = requests.post(nextjs_url, json=json_data)
    return response.status_code

@app.post("/process/")
async def process_data(background_tasks: BackgroundTasks):
    global result
    # 데이터 처리 (여기서는 단순히 딜레이를 추가)
    time.sleep(1)
    result = {"status": "completed", "data": {"message": "Processing complete"}}
    
    # 백그라운드 작업으로 Next.js에 데이터 전송
    background_tasks.add_task(send_data_to_next, result)
    
    return JSONResponse(status_code=200, content={"message": "Processing started"})

@app.get("/post_process", response_class=HTMLResponse)
async def post_process():
    html_content = """
    <html>
        <head>
            <title>Post Process</title>
        </head>
        <body>
            <h1>Post Process</h1>
            <form action="/process" method="post">
                <button type="submit">Start Process</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)