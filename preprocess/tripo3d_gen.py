import os
import asyncio
import websockets
import json
import subprocess
import requests
import argparse
import zipfile
from tqdm import tqdm

api_key = "tsk_***"

def update_json(file_path, new_items):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            original_data = json.load(file)
    else:
        original_data = {}
    
    original_data.update(new_items)

    with open(file_path, 'w') as file:
        json.dump(original_data, file, indent=4)
        
def upload_image(image_path, task_filename, log_filename, format='png'):
    url = "https://api.tripo3d.ai/v2/openapi/upload"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path, f, f'image/{format}')}
        response = requests.post(url, headers=headers, files=files)
    
    with open(log_filename, 'w') as fout:
        fout.write('Upload Image ....\n')
        json.dump(response.json(), fout, indent=4)
        fout.write('\nUpload Image Done\n')
        
    image_token = response.json()['data']['image_token']
    update_json(task_filename, {'image_type':format, 'image_token': image_token})
    
def image_to_3d_task(task_filename, log_filename):
    with open(task_filename, 'r') as f:
        task_dict = json.load(f)
        
    url = "https://api.tripo3d.ai/v2/openapi/task"
    image_type = task_dict['image_type']
    image_token = task_dict['image_token']

    data = {
        "type": "image_to_model",
        "file": {
            "type": f"{image_type}",
            "file_token": f"{image_token}"
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, json=data)

    print(response.json())
    
    with open(log_filename, 'a') as fout:
        fout.write('Image-to-3D Generation ....\n')
        json.dump(response.json(), fout, indent=4)
        fout.write('\nImage-to-3D Generation Done\n')
        
    task_id = response.json()['data']['task_id']
    update_json(task_filename, {'original_model_task_id': task_id})


def text_to_3d_task(task_filename, log_filename):
    bash_command = f'''
    curl https://api.tripo3d.ai/v2/openapi/task \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer {api_key}" \
    -d '{{"type": "text_to_model", "prompt": "a small cat"}}'
    '''
    with open(log_filename, 'w') as fout:
        fout.write(bash_command)
        fout.write('\n')
        
    result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    stdout_output = result.stdout
    print(stdout_output)
    output = json.loads(stdout_output)
    update_json(task_filename, {'original_model_task_id': output['data']['task_id']})
        
    with open(log_filename, 'a') as fout:
        fout.write(stdout_output)
        fout.write('\n')

def convert_to_obj_task(task_id, task_filename, log_filename):
    bash_command = f'''
    curl https://api.tripo3d.ai/v2/openapi/task \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer {api_key}" \
    -d '{{"type": "convert_model","format": "OBJ", "original_model_task_id": "{task_id}", "face_limit": 30000}}'
    '''
    with open(log_filename, 'a') as fout:
        fout.write(bash_command)
        fout.write('\n')
        
    result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    stdout_output = result.stdout
    print(stdout_output)
    output = json.loads(stdout_output)
    update_json(task_filename, {'convert_obj_task_id': output['data']['task_id']})
        
    with open(log_filename, 'a') as fout:
        fout.write(stdout_output)
        fout.write('\n')


async def receive_one(tid):
    url = f"wss://api.tripo3d.ai/v2/openapi/task/watch/{tid}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    async with websockets.connect(url, extra_headers=headers) as websocket:
        while True:
            message = await websocket.recv()
            try:
                data = json.loads(message)
                status = data['data']['status']
                if status not in ['running', 'queued']:
                    break
            except json.JSONDecodeError:
                print("Received non-JSON message:", message)
                break
    return data

def download_and_unzip(url, extract_to):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        zip_file_path = "temp.zip"
        with open(zip_file_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded ZIP file from {url}")

        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted ZIP file to {extract_to}")

        os.remove(zip_file_path)
        print("Temporary ZIP file removed.")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def run_single_image(img_path, save_dir, task_name, log_dir):
    if os.path.exists(os.path.join(save_dir, task_name, "tripo")):
        return
    task_filename = os.path.join(log_dir, f"{task_name}.json")
    log_filename = os.path.join(log_dir, f"{task_name}.log")
    
    """ Reconstruction """
    if not os.path.exists(task_filename):
        # text_to_3d_task(task_filename, log_filename)    
        upload_image(img_path, task_filename, log_filename)
        image_to_3d_task(task_filename, log_filename)
    with open(task_filename, 'r') as f:
        result_json = json.load(f)
    task_id = result_json['original_model_task_id']
    result = asyncio.run(receive_one(task_id))
    
    """ Convert to OBJ File """
    if 'convert_obj_task_id' not in result_json:
        convert_to_obj_task(task_id, task_filename, log_filename)
        with open(task_filename, 'r') as f:
            result_json = json.load(f) #json file was updated
            
    """ Get Conversion Result """
    task_id = result_json['convert_obj_task_id']
    result = asyncio.run(receive_one(task_id))
    model_path = result['data']['result']['model']['url']
    update_json(task_filename, {'obj_file': model_path})
    
    with open(log_filename, 'a') as fout:
        fout.write(str(result))
        fout.write('\n')
        
    download_and_unzip(model_path, os.path.join(save_dir, task_name, "tripo"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a command and save output to a specified log file.")
    parser.add_argument('--data_dir', type=str, help='Path to the data to be processed')
    parser.add_argument("--log_dir", type=str, help="Path to save log files", default="./logs/tripo3d")
    
    args = parser.parse_args()
    data_dir = args.data_dir
    log_dir = args.log_dir
    
    os.makedirs(log_dir, exist_ok=True)
    input_folder = os.path.join(data_dir, "obj_recon/input_for_lrm")
    output_folder = os.path.join(args.data_dir, "obj_recon/results/tripo/meshes/")
    for model in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, model, "full.png")
        if not os.path.exists(img_path):
            continue
        task_name = model
        save_dir = output_folder
        
        try:
            run_single_image(img_path, save_dir, task_name, log_dir)
        except Exception as ex:
            print("An error occurred:")
            print(ex)