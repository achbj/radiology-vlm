import requests
import sys

if len(sys.argv) < 4:
    print("Usage: python test_vqa.py <URL> <image_path> <question>")
    print('Example: python test_vqa.py https://my-modal-url.run test.jpg "Is there an abnormality?"')
    sys.exit(1)

url = sys.argv[1]
image_path = sys.argv[2]
question = sys.argv[3]

print(f"Sending Question: '{question}' for image {image_path} to {url}...")

try:
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"question": question}
        response = requests.post(url, files=files, data=data)

    print("\n--- Response ---")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("VLM Answer JSON:")
        import json
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:", response.text)
        
except FileNotFoundError:
    print(f"Error: The file {image_path} was not found.")
except Exception as e:
    print(f"Error: {e}")
