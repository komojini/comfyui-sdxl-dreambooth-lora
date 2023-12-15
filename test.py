import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from predict import WORKFLOW, Predictor


server_address = "192.168.1.241:8188"


def queue_prompt(prompt, client_id):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt, client_id):
    prompt_id = queue_prompt(prompt, client_id)['prompt_id']
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

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


s3_lora_url = "https://komojini-comfyui.s3.ap-northeast-2.amazonaws.com/komojini-comfyui/test/models/ms_v1.safetensors"

predictor = Predictor()

instance_prompt = "zwc"
class_prompt = "cat"
steps = 20
input_prompt = "artistic photo of 1 var_1 var_2 wearing Santa costume, small cute santa hat, Christmas tree, Christmas style, Christmas concept, (Christmas:1.2), presents, (var_1 var_2:1.3), (midnight:1.5), (fancy:1.5), twinkle, colorful background, fancy wallpaper, professional photo, 4k, profile, Christmas socks, socks"
negative_prompt = "text, watermark, low quality, day, bad body, monotone background, white wall, white background, bad hat, bad costume, 2, double hat, nsfw, bad hands"

seed = 196429611935343 

endpoint_url, s3_lora_path = predictor.split_s3_endpoint_url_and_path(s3_lora_url)

s3_access_key = ""
s3_secret_access_key = ""

workflow_string = predictor.build_workflow_string(
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


client_id = str(uuid.uuid4())
ws = websocket.WebSocket()
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
images = get_images(ws, prompt, client_id)

#Commented out code to display the output images:
for node_id in images:
    for image_data in images[node_id]:
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_data))
        image.save("out-"+node_id+".png")