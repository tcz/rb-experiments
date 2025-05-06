import argparse
import json
import subprocess
import sys
from playwright.sync_api import sync_playwright
import requests

import torch
from PIL import Image
import PIL
import torchvision.transforms as transforms
import numpy as np
import lpips

PIL.Image.MAX_IMAGE_PIXELS = 1000000000
WEB_SERVER_PORT = 8893
VALIDATION_DATA_DIR = 'data-rb-validate'

server_thread = None

def remove_alpha(image):
    if image.shape[0] == 4:
        return image[:3, :, :]
    return image

def calculate_mse(image1, image2):
    return ((image1 - image2) ** 2).mean().item()

def resize_image(image):
    return torch.nn.functional.avg_pool2d(image, 2)

def calculate_similarity(image1, image2):
    errors = []
    while True:
        mse = calculate_mse(image1, image2)

        errors.append(mse)

        _, h, w = image1.size()
        if h == 1 or w == 1:
            break

        image1 = resize_image(image1)
        image2 = resize_image(image2)

    average_mse = np.mean(errors)
    sim = 1 - average_mse

    return float(sim)

def calculate_perceptual_loss(image1, image2):
    loss_fn_alex = lpips.LPIPS(net='alex', verbose=False)
    loss_fn_alex.to(image1.device)

    # Normalize to [-1, 1]
    image1 = (image1 - 0.5) * 2
    image2 = (image2 - 0.5) * 2

    loss = loss_fn_alex(image1, image2)

    return loss.squeeze().item()

def metrics(image1_path, image2_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = transforms.ToTensor()

    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    if image1.size == image2.size:
        new_image2 = Image.new("RGB", image1.size, (255, 255, 255))
        new_image2.paste(image2, (0, 0))
        image2 = new_image2.crop((0, 0, image1.size[0], image1.size[1]))

    image1 = remove_alpha(transform(image1)).to(device)
    image2 = remove_alpha(transform(image2)).to(device)

    similarity = calculate_similarity(image1, image2)
    perceptual_loss = calculate_perceptual_loss(image1, image2)

    return {
        'similarity': similarity,
        'perceptual_loss': perceptual_loss,
    }


def take_screenshot(url, path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.screenshot(path=path)
        browser.close()

import http.server
import socketserver
import threading
import os

stop_server_flag = threading.Event()

class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=VALIDATION_DATA_DIR, **kwargs)

    def end_headers(self):
        if self.path.endswith(".html"):
            self.send_header("Content-type", "text/html")
        elif self.path.endswith(".png"):
            self.send_header("Content-type", "image/png")
        elif self.path.endswith(".jpg") or self.path.endswith(".jpeg"):
            self.send_header("Content-type", "image/jpeg")
        super().end_headers()

def start_server():
    with socketserver.TCPServer(("", WEB_SERVER_PORT), SimpleHTTPRequestHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Serving on port {WEB_SERVER_PORT}")
        while not stop_server_flag.is_set():
            httpd.serve_forever()

def calculate_metrics(predicted_markup, expected_markup):
    global server_thread

    print(expected_markup[:100])
    print(len(expected_markup))
    print(predicted_markup[:100])
    print(len(predicted_markup))

    with open(os.path.join(VALIDATION_DATA_DIR, 'predicted.html'), 'w') as file:
        file.write(predicted_markup)

    with open(os.path.join(VALIDATION_DATA_DIR, 'expected.html'), 'w') as file:
        file.write(expected_markup)

    import uuid
    predicted_screenshot_path = os.path.join(VALIDATION_DATA_DIR, f'predicted{uuid.uuid4().hex}.png')
    expected_screenshot_path = os.path.join(VALIDATION_DATA_DIR, f'expected{uuid.uuid4().hex}.png')

    try:
        if server_thread is None or not server_thread.is_alive():
            server_thread = threading.Thread(target=start_server,)
            server_thread.daemon = True
            server_thread.start()

        script_path = os.path.abspath(__file__)

        result = subprocess.run(
            [sys.executable, script_path,
                'http://127.0.0.1:' + str(WEB_SERVER_PORT) + '/predicted.html',
                'http://127.0.0.1:' + str(WEB_SERVER_PORT) + '/expected.html',
                predicted_screenshot_path,
                expected_screenshot_path
             ],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse the output as a float
        output = result.stdout.strip()

        print("Output: ", output)

        metrics = json.loads(output)
        metrics['predicted_screenshot_path'] = predicted_screenshot_path
        metrics['expected_screenshot_path'] = expected_screenshot_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Script failed with error: {e.stderr}")
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Output is not a json: {output}")

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('predicted_url', type=str,)
    parser.add_argument('expected_url', type=str,)
    parser.add_argument('predicted_screenshot_path', type=str,)
    parser.add_argument('expected_screenshot_path', type=str,)

    args = parser.parse_args()

    take_screenshot(args.predicted_url, args.predicted_screenshot_path)
    take_screenshot(args.expected_url, args.expected_screenshot_path)

    metrics = metrics(args.predicted_screenshot_path, args.expected_screenshot_path)

    print(json.dumps(metrics))

