import requests
import json

DJANGO_API_URL = "http://127.0.0.1:8000/attendance/mark/"

def send_attendance(employee_id):
    """Send attendance data to Django API."""
    data = {"employee_id": employee_id}  # ✅ Use "name" (Django expects this)

    try:
        response = requests.post(DJANGO_API_URL, json=data, headers={"Content-Type": "application/json"})
        print(f"📡 Sent Attendance for {employee_id}")
        print(f"🔍 Response Status: {response.status_code}")
        print(f"📝 Response Data: {response.text}")  # ✅ Debugging
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending attendance: {e}")
