import subprocess
import threading
import time
from cog import BasePredictor, Input, Path
from typing import List
import os
import torch
import shutil
import uuid
import json
import urllib
import websocket
import multiprocessing
from PIL import Image
from urllib.error import URLError
import random
from urllib.parse import urlparse


WORKFLOW = """
{
    "input": {
      "prompt": {
        "41": {
          "inputs": {
            "text": "_POSITIVE_PROMPT",
            "var_1": "_INSTANCE_PROMPT",
            "var_2": "_CLASS_PROMPT",
            "var_3": "",
            "var_4": "",
            "var_5": ""
          },
          "class_type": "PromptWithTemplate"
        },
        "11": {
          "inputs": {
            "lora_name": "_S3_LORA_PATH",
            "strength_model": 1,
            "strength_clip": 1,
            "BUCKET_ENDPOINT_URL": "_BUCKET_ENDPOINT_URL",
            "BUCKET_ACCESS_KEY_ID": "_BUCKET_ACCESS_KEY_ID",
            "BUCKET_SECRET_ACCESS_KEY": "_BUCKET_SECRET_ACCESS_KEY",
            "BUCKET_NAME": "_BUCKET_NAME",
            "model": [
              "4",
              0
            ],
            "clip": [
              "4",
              1
            ]
          },
          "class_type": "S3Bucket_Load_LoRA"
        },
        "5": {
          "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": "_BATCH_SIZE"
          },
          "class_type": "EmptyLatentImage"
        },
        "3": {
          "inputs": {
            "seed": "_SEED",
            "steps": "_STEPS",
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": [
              "11",
              0
            ],
            "positive": [
              "6",
              0
            ],
            "negative": [
              "7",
              0
            ],
            "latent_image": [
              "5",
              0
            ]
          },
          "class_type": "KSampler"
        },
        "4": {
          "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors"
          },
          "class_type": "CheckpointLoaderSimple"
        },
        "6": {
          "inputs": {
            "text": [
              "41",
              0
            ],
            "clip": [
              "11",
              1
            ]
          },
          "class_type": "CLIPTextEncode"
        },
        "7": {
          "inputs": {
            "text": "_NEGATIVE_PROMPT",
            "clip": [
              "11",
              1
            ]
          },
          "class_type": "CLIPTextEncode"
        },
        "8": {
          "inputs": {
            "samples": [
              "3",
              0
            ],
            "vae": [
              "4",
              2
            ]
          },
          "class_type": "VAEDecode"
        },
        "46": {
          "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
              "8",
              0
            ]
          },
          "class_type": "SaveImage"
        }
      }
    }
  }
"""


class Predictor(BasePredictor):
    def setup(self):
        # start server
        self.server_address = "127.0.0.1:8188"
        self.start_server()

    def start_server(self):
        server_thread = threading.Thread(target=self.run_server)
        server_thread.start()

        while not self.is_server_running():
            time.sleep(1)  # Wait for 1 second before checking again

        print("Server is up and running!")

    def run_server(self):
        command = "python ./ComfyUI/main.py"
        server_process = subprocess.Popen(command, shell=True)
        server_process.wait()

    # hacky solution, will fix later
    def is_server_running(self):
        try:
            with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, "123")) as response:
                return response.status == 200
        except URLError:
            return False
    
    def queue_prompt(self, prompt, client_id):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        print(folder_type)
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            return response.read()

    def get_images(self, ws, prompt, client_id):
        prompt_id = self.queue_prompt(prompt, client_id)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                continue #previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                print("node output: ", node_output)

                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append(image_data)
                output_images[node_id] = images_output

        return output_images

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())
    
    def build_workflow_string(self, **kwargs):
        new_workflow = WORKFLOW
        for key, value in kwargs.items():
            new_workflow = new_workflow.replace(key, value)
        return new_workflow
    
    def extract_region_from_url(self, endpoint_url):
        """
        Extracts the region from the endpoint URL.
        """
        parsed_url = urlparse(endpoint_url)
        # AWS/backblaze S3-like URL
        if '.s3.' in endpoint_url:
            return endpoint_url.split('.s3.')[1].split('.')[0]

        # DigitalOcean Spaces-like URL
        if parsed_url.netloc.endswith('.digitaloceanspaces.com'):
            return endpoint_url.split('.')[1].split('.digitaloceanspaces.com')[0]

        return None

    def split_s3_endpoint_url_and_path(self, s3_url):
        bucket_name = s3_url.split("/")[3]

        elements = s3_url.split(bucket_name)
        path = elements[-1]
        endpoint_url = f"{elements[0]}{bucket_name}{elements[1]}"
        return endpoint_url, path

    def get_boto_client(
            self,
            endpoint_url,
            access_key_id,
            secret_access_key
    ):
        from boto3 import session
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        bucket_session = session.Session()

        boto_config = Config(
            signature_version='s3v4',
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            }
        )

        transfer_config = TransferConfig(
            multipart_threshold=1024 * 25,
            max_concurrency=multiprocessing.cpu_count(),
            multipart_chunksize=1024 * 25,
            use_threads=True
        )

        boto_client = bucket_session.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=boto_config,
            region_name=self.extract_region_from_url(endpoint_url)
        )

        return boto_client

    # def upload_image(self, image_path, boto_client, bucket, output_dir, image_name, file_extension):
    #     file_extension = os.path.splitext(image_path)[1]
    #     content_type = "image/" + file_extension.lstrip(".")

    #     key = f"{output_dir}/{image_name}{file_extension}"
    #     boto_client.put_object(
    #         Bucket=f'{bucket}',
    #         Key=f'{output_dir}/{image_name}{file_extension}',
    #         Body=output,
    #         ContentType=content_type
    #     )


    #     presigned_url = boto_client.generate_presigned_url(
    #         'get_object',
    #         Params={
    #             'Bucket': f'{bucket}',
    #             'Key': f'{job_id}/{image_name}{file_extension}'
    #         }, ExpiresIn=604800)

    def predict(
        self,
        input_prompt: str = Input(description="Prompt", default="A photo of var_1 var_2"),
        negative_prompt: str = Input(description="Negative Prompt", default="text, watermark, ugly, blurry"),
        steps: int = Input(
            description="Steps",
            default=20
        ),
        instance_prompt: str = Input(description="Instance Prompt (var_1)", default="cat"),
        class_prompt: str = Input(description="Class Prompt (var_2)", default="zwc"),
        seed: int = Input(description="Sampling seed, leave Empty for Random", default=None),
        s3_lora_url: str = Input(description="S3 LoRA Model URL", default="https://<bucket-name>.s3.<region>.amazonaws.com/<path-to-lora-model>.safetensors"),
        s3_access_key: str = Input(description="S3 Access Key", default=None),
        s3_secret_access_key: str = Input(description="S3 Secret Access Key", default=None),
        s3_output_dir: str = Input(description="S3 Image Save Directory", default=None)
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        endpoint_url, s3_lora_path = self.split_s3_endpoint_url_and_path(s3_lora_url)
        
        workflow_string = self.build_workflow_string(
            _POSITIVE_PROMPT = input_prompt,
            _NEGATIVE_PROMPT = negative_prompt,
            _STEPS = steps,
            _SEED = seed,
            _INSTANCE_PROMPT = instance_prompt,
            _CLASS_PROMPT = class_prompt,
            _S3_LORA_PATH = s3_lora_path,
            _BUCKET_ENDPOINT_URL = endpoint_url,
            _BUCKET_ACCESS_KEY_ID = s3_access_key,
            _BUCKET_SECRET_ACCESS_KEY = s3_secret_access_key 
        )
        # load config
        prompt = json.loads(workflow_string)

        if not prompt:
            raise Exception('no workflow config found')

        # start the process
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, client_id))
        images = self.get_images(ws, prompt, client_id)

        image_paths = []
        for node_id in images:
            for image_data in images[node_id]:
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                image.save("out-"+node_id+".png")
                image_paths.append(Path("out-"+node_id+".png"))
        
        if s3_output_dir:
            boto_client = self.get_boto_client(endpoint_url, s3_access_key, s3_secret_access_key)
            for image_path in image_paths:
                pass

        return image_paths