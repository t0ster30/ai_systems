import requests

try:
    # Replace with the URL that is failing
    requests.get(r"https://pypi.org", verify="C:\\Users\\29415403\\Documents\\certs.pem")
    print("Success! The bundle is working.")
except Exception as e:
    print(f"Validation failed: {e}")
