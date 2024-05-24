from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import aiofiles
import asyncio
import zipfile
import os
import pandas as pd
import base64
import requests
from docker_utils import docker_cp, docker_exec, get_container_files
from ocr_crop import crop_images
import random
import json

app = FastAPI()

progress = 0

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except RuntimeError as e:
                print(f"Error sending message: {e}")
                self.active_connections.remove(connection)

manager = ConnectionManager()

async def update_progress(increment: int):
    global progress
    progress = increment
    if progress > 100:
        progress = 100
    await manager.send_message(json.dumps({"progress": progress}))

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # Keep the connection alive
            await websocket.send_json({"progress": progress})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def run_crop_images(image_path: str, save_dir: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, crop_images, image_path, save_dir)

async def create_zip_file(cropped_image_paths: list, zip_file_path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, zip_cropped_images, cropped_image_paths, zip_file_path)

def zip_cropped_images(cropped_image_paths: list, zip_file_path: str):
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        for file_path in cropped_image_paths:
            zipf.write(file_path, os.path.basename(file_path))
            
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def send_data_to_next(json_data):
    # Next.js 서버의 엔드포인트로 JSON 데이터를 POST 요청으로 전송
    nextjs_url = "http://localhost:3000/api/receive"
    response = requests.post(nextjs_url, json=json_data)
    return response.status_code

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    global progress
    progress = 0
    try:
        # Save the uploaded file to a temporary directory
        file_location = f"./tmp/{file.filename}"
        
        async with aiofiles.open(file_location, "wb") as f:
            contents = await file.read()
            await f.write(contents)
        file_location = f"./tmp/bbb.jpg"
        
        # Perform OCR cropping and get the list of cropped image paths
        print("Starting OCR cropping...")
        cropped_image_paths, detected_image_path = crop_images(file_location, 'saves')
        print(f"Cropped image paths: {cropped_image_paths}")
        print(f"Detected image path: {detected_image_path}")

        # Update progress
        await update_progress(10)

        # Create a zip file of the cropped images
        zip_file_path = "./tmp/cropped_images.zip"
        print("Creating zip file...")
        await create_zip_file(cropped_image_paths, zip_file_path)
        print(f"Zip file created at: {zip_file_path}")

        # Update progress
        await update_progress(20)

        # Define Docker container and paths
        container_ocr_name = "elegant_goldwasser"
        container_ocr_zip_path = "/letr/data/data/real_testset/cropped_images.zip"
        container_ocr_unzip_dir = "/letr/data/data/real_testset/"
        container_ocr_results_dir = "/letr/data/result/"
        host_results_dir = "./result/"
        
        container_translate_name = "serene_shaw"
        container_translate_custom_dir = "/letr/data/custom/"
        container_translate_custom_result_dir = "/letr/data/custom/result/"
        container_translate_custom_data_dir = "/letr/data/custom/data"
        
        # Copy the zip file to the Docker container
        print(f"Copying zip file to Docker container: {container_ocr_name}")
        copy_result = await docker_cp(zip_file_path, f"{container_ocr_name}:{container_ocr_zip_path}")
        if copy_result.returncode != 0:
            print(f"Failed to copy zip file to container: {copy_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to copy zip file to container", "details": copy_result.stderr.decode()})

        # Update progress
        await update_progress(30)

        # Unzip the file inside the Docker container
        print(f"Unzipping file in Docker container: {container_ocr_name}")
        unzip_command = f"unzip -o {container_ocr_zip_path} -d {container_ocr_unzip_dir}"
        unzip_result = await docker_exec(container_ocr_name, unzip_command)
        if unzip_result.returncode != 0:
            print(f"Failed to unzip file in container: {unzip_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to unzip file in container", "details": unzip_result.stderr.decode()})

        # Update progress
        await update_progress(40)

        # Execute the Python script inside the Docker container
        print(f"Executing script in Docker container: {container_ocr_name}")
        script_path = "/letr/inferences/text_recognition.py"
        exec_command = (
            f"python {script_path} "
            f"--Transformation TPS --FeatureExtraction VGG --SequenceModeling BiLSTM "
            f"--Prediction Attn --image_folder {container_ocr_unzip_dir} "
            f"--saved_model saved_models/TPS-VGG-BiLSTM-Attn-final/best_accuracy.pth "
            f"--batch_max_length 50 --workers 16 --batch_size 64 --imgH 64 --imgW 300"
        )
        exec_result = await docker_exec(container_ocr_name, exec_command)
        if exec_result.returncode != 0:
            print(f"Failed to process file: {exec_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to process file", "details": exec_result.stderr.decode()})

        print(f"Script executed successfully: {exec_result.stdout.decode()}")

        # Update progress
        await update_progress(50)

        # Find the results file in the Docker container
        print(f"Finding results file in Docker container: {container_ocr_name}")
        results_files = await get_container_files(container_ocr_name, container_ocr_results_dir)
        xlsx_files = [f for f in results_files if f.endswith('.xlsx')]
        if not xlsx_files:
            print("No Excel files found in the results directory.")
            return JSONResponse(status_code=500, content={"message": "No Excel files found in the results directory."})

        container_results_path = os.path.join(container_ocr_results_dir, xlsx_files[0])
        host_results_path = os.path.join(host_results_dir, xlsx_files[0])
        
        # Copy the results file from the Docker container to the host
        os.makedirs(host_results_dir, exist_ok=True)
        print(f"Copying results file from Docker container: {container_ocr_name}")
        copy_results_result = await docker_cp(f"{container_ocr_name}:{container_results_path}", host_results_path)
        if copy_results_result.returncode != 0:
            print(f"Failed to copy results file from container: {copy_results_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to copy results file from container", "details": copy_results_result.stderr.decode()})

        # Update progress
        await update_progress(60)

        # Copy the results file from the host to the second Docker container
        print(f"Copying results file to Docker container: {container_translate_name}")
        copy_to_second_container_result = await docker_cp(host_results_path, f"{container_translate_name}:{container_translate_custom_data_dir}")
        if copy_to_second_container_result.returncode != 0:
            print(f"Failed to copy results file to second container: {copy_to_second_container_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to copy results file to second container", "details": copy_to_second_container_result.stderr.decode()})

        # Update progress
        await update_progress(70)

        # Execute the process_excel.py script inside the second Docker container
        print(f"Executing process_excel.py in Docker container: {container_translate_name}")
        exec_command = f"python data/custom/process_excel.py"
        exec_result = await docker_exec(container_translate_name, exec_command)
        if exec_result.returncode != 0:
            print(f"Failed to execute process_excel.py: {exec_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to execute process_excel.py", "details": exec_result.stderr.decode()})

        print(f"process_excel.py executed successfully: {exec_result.stdout.decode()}")

        # Update progress
        await update_progress(80)

        # Execute the translate.py script inside the second Docker container
        print(f"Executing translate.py in Docker container: {container_translate_name}")
        translate_command = (
            "python translate.py -src_lang ko -tgt_lang en "
            "-src data/custom/result/output_file.csv "
            "-model data/models/en_model.pt "
            "-output data/custom/result --report_time"
        )
        translate_result = await docker_exec(container_translate_name, translate_command)
        if translate_result.returncode != 0:
            print(f"Failed to execute translate.py: {translate_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to execute translate.py", "details": translate_result.stderr.decode()})

        print(f"translate.py executed successfully: {translate_result.stdout.decode()}")

        # Update progress
        await update_progress(90)

        # Find the results file in the Docker container
        print(f"Finding results file in Docker container: {container_translate_name}")
        results_files = await get_container_files(container_translate_name, container_translate_custom_result_dir)
        xlsx_files = [f for f in results_files if f.endswith('.xlsx')]
        if not xlsx_files:
            print("No Excel files found in the results directory.")
            return JSONResponse(status_code=500, content={"message": "No Excel files found in the results directory."})

        host_results_dir = "./result/"
        os.makedirs(host_results_dir, exist_ok=True)

        # Copy the results file from the Docker container to the host
        for xlsx_file in xlsx_files:
            container_results_path = os.path.join(container_translate_custom_result_dir, xlsx_file)
            host_results_path = os.path.join(host_results_dir, xlsx_file)
            print(f"Copying results file from Docker container: {container_translate_name}")
            copy_results_result = await docker_cp(f"{container_translate_name}:{container_results_path}", host_results_path)
            if copy_results_result.returncode != 0:
                print(f"Failed to copy results file from container: {copy_results_result.stderr.decode()}")
                return JSONResponse(status_code=500, content={"message": "Failed to copy results file from container", "details": copy_results_result.stderr.decode()})

        # Load and process the final results file
        excel_path = os.path.join(host_results_dir, os.listdir(host_results_dir)[0])
        df = pd.read_excel(excel_path)
        text_data = {}
        for index, row in df.iterrows():
            row_data = {}
            for col in ['ko', 'prediction']:
                if pd.notna(row[col]):
                    row_data[col] = row[col]
            text_data[str(index)] = row_data    

        # Encode images to Base64
        original_image_base64 = encode_image_to_base64(file_location)
        detected_image_base64 = encode_image_to_base64(detected_image_path)
        cropped_images_base64 = [encode_image_to_base64(image_path) for image_path in cropped_image_paths]
        
        response_data = {
            "original": original_image_base64,
            "detect": detected_image_base64,
            "crop": cropped_images_base64,
            "data": text_data
        }
        
        # Optionally, you can return a response or additional processing here
        send_status = send_data_to_next(response_data)
        if send_status != 200:
            return JSONResponse(status_code=500, content={"message": "Failed to send data to Next.js"})

        # 마지막 단계에서 100% 설정
        await update_progress(100)
        return JSONResponse(status_code=200, content={"message": "File processed and translated successfully."})     
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "An error occurred", "details": str(e)})