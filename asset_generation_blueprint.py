#!/usr/bin/env python3
"""
Asset Generation Blueprint Interactive Script

This script uses the Stability AI API to generate various assets:
  - Image generation (default)
  - Video generation
  - Standard 3D model generation
  - Sketch-to-Image (refine a sketch)
  - SD3 image generation
  - SD3.5 image generation
  - Upscaling an image
  - 3D Aware generation

The script is interactive:
  - The user is prompted for a mode (with a default).
  - For the chosen mode, any mandatory parameters are requested.
  - Default values are provided for optional parameters.
  - Errors (missing input, invalid values) are printed as needed.

Environment:
  - The API key is loaded from the .env file (as STABILITY_API_KEY).
  - Make sure the API key is set before running this script.
  
Author: Updated with added endpoints, models, and unique file naming  
"""

from dotenv import load_dotenv
import os
import time
import requests

# Load environment variables, including STABILITY_API_KEY.
load_dotenv()
api_key = os.getenv("STABILITY_API_KEY")
if not api_key:
    print("Error: 'STABILITY_API_KEY' not found. Please set it in your .env file.")
    exit(1)

# -------------------------------------------------------------------
# Global Endpoint URLs for various asset generation modes.
# -------------------------------------------------------------------
IMAGE_GENERATION_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
VIDEO_GENERATION_ENDPOINT = "https://api.stability.ai/v2beta/image-to-video"
THREE_D_GENERATION_ENDPOINT = "https://api.stability.ai/v2beta/3d/stable-fast-3d"
SKETCH_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/control/sketch"
SD3_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
UPSCALER_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/upscale/creative"
THREE_D_AWARE_ENDPOINT = "https://api.stability.ai/v2beta/3d/stable-point-aware-3d"

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def unique_filename(prefix, ext, seed=None):
    """
    Generate a unique filename by appending the current timestamp to the prefix.
    Optionally include the seed if provided.
    """
    ts = int(time.time())
    if seed is not None:
        return f"{prefix}_{seed}_{ts}.{ext}"
    return f"{prefix}_{ts}.{ext}"

def send_generation_request(host, params, files=None, api_key=None):
    """
    Send a synchronous API request to the given endpoint.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }
    if files is None:
        files = {}
    # If a file is referenced by 'image' or 'mask', open it.
    image_file = params.pop("image", None)
    mask_file = params.pop("mask", None)
    if image_file and os.path.exists(image_file):
        files["image"] = open(image_file, 'rb')
    if mask_file and os.path.exists(mask_file):
        files["mask"] = open(mask_file, 'rb')
    if not files:
        files["none"] = ("", "")

    print(f"Sending REST request to {host}...")
    response = requests.post(host, headers=headers, files=files, data=params)
    
    # Close any opened file objects.
    for file in files.values():
        if hasattr(file, "close"):
            file.close()
    
    if not response.ok:
        print(f"Error: HTTP {response.status_code}: {response.text}")
        exit(1)
    
    return response

def send_async_generation_request(host, params, files=None, api_key=None, poll_interval=10, timeout=500):
    """
    Send an asynchronous API request and poll until the result is available.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if files is None:
        files = {}
    image_file = params.pop("image", None)
    mask_file = params.pop("mask", None)
    if image_file and os.path.exists(image_file):
        files["image"] = open(image_file, 'rb')
    if mask_file and os.path.exists(mask_file):
        files["mask"] = open(mask_file, 'rb')
    if not files:
        files["none"] = ("", "")

    print(f"Sending asynchronous request to {host}...")
    response = requests.post(host, headers=headers, files=files, data=params)
    for file in files.values():
        if hasattr(file, "close"):
            file.close()
    
    if not response.ok:
        print(f"Error: HTTP {response.status_code}: {response.text}")
        exit(1)
    
    response_dict = response.json()
    generation_id = response_dict.get("id")
    if generation_id is None:
        print("Error: Expected an 'id' in the response")
        exit(1)
    
    poll_url = f"https://api.stability.ai/v2beta/results/{generation_id}"
    start = time.time()
    while True:
        print(f"Polling results at {poll_url}...")
        poll_response = requests.get(
            poll_url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "*/*"}
        )
        if not poll_response.ok:
            print(f"Error: HTTP {poll_response.status_code}: {poll_response.text}")
            exit(1)
        if poll_response.status_code != 202:
            break
        if time.time() - start > timeout:
            print(f"Error: Timeout after {timeout} seconds")
            exit(1)
        time.sleep(poll_interval)
    
    return poll_response

# -------------------------------------------------------------------
# Generation Functions for each mode.
# -------------------------------------------------------------------
def generate_image(api_key, prompt, negative_prompt="", aspect_ratio="3:2",
                   seed=42, output_format="jpeg", endpoint=IMAGE_GENERATION_ENDPOINT):
    """
    Generate an image using a text prompt.
    Returns the filename of the saved image.
    """
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "output_format": output_format,
    }
    try:
        response = send_generation_request(endpoint, params, api_key=api_key)
    except Exception as e:
        print(f"Error during image generation: {e}")
        exit(1)
    
    finish_reason = response.headers.get("finish-reason", "")
    if finish_reason == "CONTENT_FILTERED":
        print("Error: Generation failed NSFW classifier.")
        exit(1)
    
    filename = unique_filename("generated", output_format, seed)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"Image successfully generated and saved as '{filename}'.")
    return filename

def generate_video(api_key, image_path, seed=42, cfg_scale=7.5, motion_bucket_id=127,
                   endpoint=VIDEO_GENERATION_ENDPOINT):
    """
    Generate a video from an image.
    Returns the filename of the saved video.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"Sending video generation request to {endpoint}...")
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                endpoint,
                headers=headers,
                files={"image": image_file},
                data={
                    "seed": seed,
                    "cfg_scale": cfg_scale,
                    "motion_bucket_id": motion_bucket_id,
                }
            )
    except Exception as e:
        print(f"Error reading image file: {e}")
        exit(1)
    
    if not response.ok:
        print(f"Error: Video generation failed: HTTP {response.status_code}: {response.text}")
        exit(1)
    
    generation_id = response.json().get("id")
    if not generation_id:
        print("Error: No generation ID received.")
        exit(1)
    
    result_url = f"https://api.stability.ai/v2beta/image-to-video/result/{generation_id}"
    while True:
        print("Polling video generation result...")
        poll_response = requests.get(
            result_url,
            headers={"Authorization": f"Bearer {api_key}", "accept": "video/*"}
        )
        if poll_response.status_code == 202:
            print("Video generation in progress, waiting 10 seconds...")
            time.sleep(10)
            continue
        elif poll_response.status_code == 200:
            filename = unique_filename("video", "mp4", seed)
            with open(filename, "wb") as f:
                f.write(poll_response.content)
            print(f"Video successfully generated and saved as '{filename}'.")
            return filename
        else:
            print(f"Error: {poll_response.text}")
            exit(1)

def generate_3d(api_key, image_path, texture_resolution="1024", foreground_ratio=0.85,
                remesh="triangle", vertex_count=20000, additional_params=None,
                endpoint=THREE_D_GENERATION_ENDPOINT):
    """
    Generate a 3D model (GLB) from an image.
    Returns the filename of the saved 3D model.
    """
    params = {
        "texture_resolution": texture_resolution,
        "foreground_ratio": foreground_ratio,
        "remesh": remesh,
        "vertex_count": vertex_count,
        "image": image_path,
    }
    if additional_params:
        params.update(additional_params)
    
    try:
        response = send_generation_request(endpoint, params, api_key=api_key)
    except Exception as e:
        print(f"Error during 3D model generation: {e}")
        exit(1)
    
    filename = unique_filename("model", "glb")
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"3D model successfully generated and saved as '{filename}'.")
    return filename

def sketch_to_image(api_key, input_image, prompt, negative_prompt="", control_strength=0.7,
                    seed=42, output_format="jpeg", endpoint=SKETCH_ENDPOINT):
    """
    Transform a sketch (input image) into a refined output image.
    Returns the filename of the saved image.
    """
    if not os.path.exists(input_image):
        print("Error: The specified input file does not exist.")
        exit(1)

    params = {
        "control_strength": control_strength,
        "seed": seed,
        "output_format": output_format,
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }

    try:
        with open(input_image, "rb") as image_file:
            files = {"image": image_file}
            print(f"Sending sketch-to-image request to {endpoint}...")
            response = requests.post(endpoint, headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "image/*"
            }, files=files, data=params)
    except Exception as e:
        print(f"Error during sketch-to-image generation: {e}")
        exit(1)

    if not response.ok:
        print(f"Error: HTTP {response.status_code}: {response.text}")
        exit(1)

    # Decode response
    finish_reason = response.headers.get("finish-reason", "")
    if finish_reason == "CONTENT_FILTERED":
        print("Error: Generation failed NSFW classifier.")
        exit(1)

    # Save the generated image
    filename, _ = os.path.splitext(os.path.basename(input_image))
    edited = f"edited_{filename}_{seed}.{output_format}"
    with open(edited, "wb") as f:
        f.write(response.content)
    print(f"Sketch-to-image generated and saved as '{edited}'.")
    return edited

def generate_sd3(api_key, prompt, negative_prompt="", aspect_ratio="3:2",
                 seed=42, output_format="jpeg", model="sd3.5-large", endpoint=SD3_ENDPOINT):
    """
    Generate an image using the SD3 or SD3.5 model.
    Returns the filename of the saved image.
    """
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt if model == "sd3" else "",
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "output_format": output_format,
        "model": model,
        "mode": "text-to-image"
    }
    try:
        response = send_generation_request(endpoint, params, api_key=api_key)
    except Exception as e:
        print(f"Error during SD3 image generation: {e}")
        exit(1)

    # Decode response
    finish_reason = response.headers.get("finish-reason", "")
    if finish_reason == "CONTENT_FILTERED":
        print("Error: Generation failed NSFW classifier.")
        exit(1)

    filename = unique_filename("sd3_generated", output_format, seed)
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"SD3 image generated and saved as '{filename}'.")
    return filename

def upscale_image(api_key, input_image, prompt, negative_prompt="", seed=42, creativity=0.3,
                  output_format="jpeg", endpoint=UPSCALER_ENDPOINT):
    """
    Upscale an image using the creative upscaler.
    Returns the filename of the saved upscaled image.
    """
    if not os.path.exists(input_image):
        print("Error: The specified input file does not exist.")
        exit(1)

    # Check if the image dimensions are within the allowed limit
    from PIL import Image
    with Image.open(input_image) as img:
        width, height = img.size
        total_pixels = width * height
        if total_pixels > 1048576:  # 1,048,576 pixels (e.g., 1024x1024)
            print(f"Error: Image dimensions exceed the maximum allowed pixels (1,048,576). Current: {total_pixels} pixels.")
            exit(1)

    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "creativity": creativity,
        "output_format": output_format
    }

    try:
        with open(input_image, "rb") as image_file:
            files = {"image": image_file}
            print(f"Sending upscale request to {endpoint}...")
            response = send_async_generation_request(endpoint, params, files=files, api_key=api_key)
    except Exception as e:
        print(f"Error during image upscaling: {e}")
        exit(1)

    if not response.ok:
        print(f"Error: HTTP {response.status_code}: {response.text}")
        exit(1)

    # Decode response
    finish_reason = response.headers.get("finish-reason", "")
    if finish_reason == "CONTENT_FILTERED":
        print("Error: Generation failed NSFW classifier.")
        exit(1)

    # Save the upscaled image
    filename, _ = os.path.splitext(os.path.basename(input_image))
    upscaled = f"upscaled_{filename}_{seed}.{output_format}"
    with open(upscaled, "wb") as f:
        f.write(response.content)
    print(f"Upscaled image saved as '{upscaled}'.")
    return upscaled

def generate_3d_aware(api_key, image_path, texture_resolution="1024",
                      foreground_ratio=0.85, remesh="triangle", vertex_count=20000,
                      additional_params=None, endpoint=THREE_D_AWARE_ENDPOINT):
    """
    Generate a 3D model (GLB) using a 3D aware method from an input image.
    Returns the filename of the saved 3D model.
    """
    params = {
        "texture_resolution": texture_resolution,
        "foreground_ratio": foreground_ratio,
        "remesh": remesh,
        "vertex_count": vertex_count,
        "image": image_path,
    }
    if additional_params:
        params.update(additional_params)
    
    try:
        response = send_generation_request(endpoint, params, api_key=api_key)
    except Exception as e:
        print(f"Error during 3D aware generation: {e}")
        exit(1)
    
    filename = unique_filename("model_3d_aware", "glb")
    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"3D aware model generated and saved as '{filename}'.")
    return filename

# -------------------------------------------------------------------
# Interactive Main Function
# -------------------------------------------------------------------
def interactive_main():
    print("Welcome to the Stability AI Asset Generation Script")
    print("Press Enter to accept the default values shown in [brackets].\n")
    
    print("Supported modes:")
    print("  img        -> Standard image generation")
    print("  video      -> Video generation from an image")
    print("  3d         -> Standard 3D model generation")
    print("  sketch     -> Sketch-to-image (refine a sketch)")
    print("  sd3        -> SD3 image generation")
    print("  upscale    -> Upscale an image")
    print("  3d-aware   -> 3D aware model generation")
    
    mode = input("Choose mode [img]: ").strip().lower() or "img"

    if mode == "img":
        prompt = input("Enter the prompt for image generation (required): ").strip()
        if not prompt:
            print("Error: A text prompt is required for image generation.")
            exit(1)
        negative_prompt = input("Enter a negative prompt [default: '']: ").strip() or ""
        aspect_ratio = input("Enter aspect ratio [default: 3:2]: ").strip() or "3:2"
        seed_str = input("Enter seed [default: 42]: ").strip() or "42"
        try:
            seed = int(seed_str)
        except ValueError:
            print("Error: Seed must be an integer.")
            exit(1)
        output_format = input("Enter output format [default: jpeg]: ").strip() or "jpeg"
        
        generate_image(
            api_key=api_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            output_format=output_format
        )
    
    elif mode == "video":
        input_file = input("Enter the source image path for video generation (required): ").strip()
        if not input_file or not os.path.exists(input_file):
            print("Error: A valid input file is required for video generation.")
            exit(1)
        seed_str = input("Enter seed [default: 42]: ").strip() or "42"
        try:
            seed = int(seed_str)
        except ValueError:
            print("Error: Seed must be an integer.")
            exit(1)
        cfg_scale_str = input("Enter CFG scale [default: 7.5]: ").strip() or "7.5"
        try:
            cfg_scale = float(cfg_scale_str)
        except ValueError:
            print("Error: CFG scale must be a number.")
            exit(1)
        motion_bucket_str = input("Enter motion bucket ID [default: 127]: ").strip() or "127"
        try:
            motion_bucket_id = int(motion_bucket_str)
        except ValueError:
            print("Error: Motion bucket ID must be an integer.")
            exit(1)
        
        generate_video(
            api_key=api_key,
            image_path=input_file,
            seed=seed,
            cfg_scale=cfg_scale,
            motion_bucket_id=motion_bucket_id
        )

    elif mode == "3d":
        input_file = input("Enter the source image path for 3D generation (required): ").strip()
        if not input_file or not os.path.exists(input_file):
            print("Error: A valid input file is required for 3D generation.")
            exit(1)
        texture_resolution = input("Enter texture resolution [default: 1024]: ").strip() or "1024"
        foreground_ratio_str = input("Enter foreground ratio [default: 0.85]: ").strip() or "0.85"
        try:
            foreground_ratio = float(foreground_ratio_str)
        except ValueError:
            print("Error: Foreground ratio must be a number.")
            exit(1)
        remesh = input("Enter remesh option [default: triangle]: ").strip() or "triangle"
        vertex_count_str = input("Enter vertex count [default: 20000]: ").strip() or "20000"
        try:
            vertex_count = int(vertex_count_str)
        except ValueError:
            print("Error: Vertex count must be an integer.")
            exit(1)
        
        generate_3d(
            api_key=api_key,
            image_path=input_file,
            texture_resolution=texture_resolution,
            foreground_ratio=foreground_ratio,
            remesh=remesh,
            vertex_count=vertex_count
        )

    elif mode == "sketch":
        input_file = input("Enter the sketch image path (required): ").strip()
        if not input_file or not os.path.exists(input_file):
            print("Error: A valid sketch image file is required.")
            exit(1)
        prompt = input("Enter the prompt for sketch-to-image generation (required): ").strip()
        if not prompt:
            print("Error: A text prompt is required for sketch-to-image generation.")
            exit(1)
        negative_prompt = input("Enter a negative prompt [default: '']: ").strip() or ""
        control_strength_str = input("Enter control strength [default: 0.7]: ").strip() or "0.7"
        try:
            control_strength = float(control_strength_str)
            if not (0 <= control_strength <= 1):
                print("Error: Control strength must be between 0 and 1.")
                exit(1)
        except ValueError:
            print("Error: Control strength must be a number.")
            exit(1)
        seed_str = input("Enter seed [default: 42]: ").strip() or "42"
        try:
            seed = int(seed_str)
        except ValueError:
            print("Error: Seed must be an integer.")
            exit(1)
        output_format = input("Enter output format [default: jpeg]: ").strip() or "jpeg"

        sketch_to_image(
            api_key=api_key,
            input_image=input_file,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_strength=control_strength,
            seed=seed,
            output_format=output_format
        )

    elif mode == "sd3":
        prompt = input("Enter the prompt for SD3 image generation (required): ").strip()
        if not prompt:
            print("Error: A text prompt is required for SD3 generation.")
            exit(1)
        negative_prompt = input("Enter a negative prompt [default: '']: ").strip() or ""
        aspect_ratio = input("Enter aspect ratio [default: 3:2]: ").strip() or "3:2"
        seed_str = input("Enter seed [default: 42]: ").strip() or "42"
        try:
            seed = int(seed_str)
        except ValueError:
            print("Error: Seed must be an integer.")
            exit(1)
        output_format = input("Enter output format [default: jpeg]: ").strip() or "jpeg"
        print("Choose a model:")
        print("  1 -> sd3.5-large")
        print("  2 -> sd3-large-turbo")
        print("  3 -> sd3-medium")
        model_choice = input("Enter your choice [1]: ").strip() or "1"
        model_map = {
            "1": "sd3.5-large",
            "2": "sd3-large-turbo",
            "3": "sd3-medium"
        }
        model = model_map.get(model_choice, "sd3.5-large")

        generate_sd3(
            api_key=api_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            output_format=output_format,
            model=model
        )

    

    elif mode == "upscale":
        input_file = input("Enter the image path to upscale (required): ").strip()
        if not input_file or not os.path.exists(input_file):
            print("Error: A valid input file is required for upscaling.")
            exit(1)
        prompt = input("Enter the prompt for upscaling (required): ").strip()
        if not prompt:
            print("Error: A prompt is required for upscaling.")
            exit(1)
        negative_prompt = input("Enter a negative prompt [default: '']: ").strip() or ""
        seed_str = input("Enter seed [default: 42]: ").strip() or "42"
        try:
            seed = int(seed_str)
        except ValueError:
            print("Error: Seed must be an integer.")
            exit(1)
        creativity_str = input("Enter creativity level [default: 0.3]: ").strip() or "0.3"
        try:
            creativity = float(creativity_str)
            if not (0 <= creativity <= 1):
                print("Error: Creativity must be between 0 and 1.")
                exit(1)
        except ValueError:
            print("Error: Creativity must be a number.")
            exit(1)
        output_format = input("Enter output format [default: jpeg]: ").strip() or "jpeg"

        upscale_image(
            api_key=api_key,
            input_image=input_file,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            creativity=creativity,
            output_format=output_format
        )

    elif mode == "3d-aware":
        input_file = input("Enter the source image path for 3D aware generation (required): ").strip()
        if not input_file or not os.path.exists(input_file):
            print("Error: A valid input file is required for 3D aware generation.")
            exit(1)
        texture_resolution = input("Enter texture resolution [default: 1024]: ").strip() or "1024"
        foreground_ratio_str = input("Enter foreground ratio [default: 1.0]: ").strip() or "1.0"
        try:
            foreground_ratio = float(foreground_ratio_str)
            if foreground_ratio < 1.0:
                print("Error: Foreground ratio must be greater than or equal to 1.")
                exit(1)
        except ValueError:
            print("Error: Foreground ratio must be a number.")
            exit(1)
        remesh = input("Enter remesh option [default: triangle]: ").strip() or "triangle"
        vertex_count_str = input("Enter vertex count [default: 20000]: ").strip() or "20000"
        try:
            vertex_count = int(vertex_count_str)
        except ValueError:
            print("Error: Vertex count must be an integer.")
            exit(1)
        
        generate_3d_aware(
            api_key=api_key,
            image_path=input_file,
            texture_resolution=texture_resolution,
            foreground_ratio=foreground_ratio,
            remesh=remesh,
            vertex_count=vertex_count
        )

    else:
        print("Error: Invalid mode specified. Please choose one of the supported modes.")
        exit(1)

if __name__ == "__main__":
    interactive_main()
