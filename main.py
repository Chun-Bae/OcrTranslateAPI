from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse,FileResponse
import aiofiles
import asyncio
import zipfile
import os
from docker_utils import docker_cp, docker_exec, get_container_files
from ocr_crop import crop_images

app = FastAPI()

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

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        file_location = f"./tmp/{file.filename}"
        
        async with aiofiles.open(file_location, "wb") as f:
            contents = await file.read()
            await f.write(contents)
            
        file_location = f"./tmp/bbb.jpg"
        # Perform OCR cropping and get the list of cropped image paths
        print("Starting OCR cropping...")
        cropped_image_paths = await run_crop_images(file_location, 'saves')
        print(f"Cropped image paths: {cropped_image_paths}")

        # Create a zip file of the cropped images
        zip_file_path = "./tmp/cropped_images.zip"
        print("Creating zip file...")
        await create_zip_file(cropped_image_paths, zip_file_path)
        print(f"Zip file created at: {zip_file_path}")

        # Define Docker container and paths
        container_ocr_name = "elegant_goldwasser"
        container_ocr_zip_path = "/letr/data/data/real_testset/cropped_images.zip"
        container_ocr_unzip_dir = "/letr/data/data/real_testset/"
        container_ocr_results_dir = "/letr/data/result/"
        host_results_dir = "./result/"
        
        container_translate_name = "serene_shaw"
        container_translate_custom_dir = "/letr/data/custom/"
        container_translate_custom_data_dir = "/letr/data/custom/data"
        
        # Copy the zip file to the Docker container
        print(f"Copying zip file to Docker container: {container_ocr_name}")
        copy_result = await docker_cp(zip_file_path, f"{container_ocr_name}:{container_ocr_zip_path}")
        if copy_result.returncode != 0:
            print(f"Failed to copy zip file to container: {copy_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to copy zip file to container", "details": copy_result.stderr.decode()})

        # Unzip the file inside the Docker container
        print(f"Unzipping file in Docker container: {container_ocr_name}")
        unzip_command = f"unzip {container_ocr_zip_path} -d {container_ocr_unzip_dir}"
        unzip_result = await docker_exec(container_ocr_name, unzip_command)
        if unzip_result.returncode != 0:
            print(f"Failed to unzip file in container: {unzip_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to unzip file in container", "details": unzip_result.stderr.decode()})

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

        # Copy the results file from the host to the second Docker container
        print(f"Copying results file to Docker container: {container_translate_name}")
        copy_to_second_container_result = await docker_cp(host_results_path, f"{container_translate_name}:{container_translate_custom_data_dir}")
        copy_to_second_container_result = await docker_cp(host_results_path, f"{container_translate_name}:{container_translate_custom_data_dir}")
        if copy_to_second_container_result.returncode != 0:
            print(f"Failed to copy results file to second container: {copy_to_second_container_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to copy results file to second container", "details": copy_to_second_container_result.stderr.decode()})

                # Execute the process_excel.py script inside the second Docker container
        print(f"Executing process_excel.py in Docker container: {container_translate_name}")
        exec_command = f"python data/custom/process_excel.py"
        exec_result = await docker_exec(container_translate_name, exec_command)
        if exec_result.returncode != 0:
            print(f"Failed to execute process_excel.py: {exec_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to execute process_excel.py", "details": exec_result.stderr.decode()})

        print(f"process_excel.py executed successfully: {exec_result.stdout.decode()}")

        # Execute the translate.py script inside the second Docker container
        print(f"Executing translate.py in Docker container: {container_translate_name}")
        translate_command = (
            "python translate.py -src_lang ko -tgt_lang en "
            "-src data/custom/data/output_file.csv "
            "-model data/models/en_model.pt "
            "-output data/custom/result --report_time"
        )
        translate_result = await docker_exec(container_translate_name, translate_command)
        if translate_result.returncode != 0:
            print(f"Failed to execute translate.py: {translate_result.stderr.decode()}")
            return JSONResponse(status_code=500, content={"message": "Failed to execute translate.py", "details": translate_result.stderr.decode()})

        print(f"translate.py executed successfully: {translate_result.stdout.decode()}")

        # Optionally, you can return a response or additional processing here
        return JSONResponse(status_code=200, content={"message": "File processed and translated successfully."})
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "An error occurred", "details": str(e)})