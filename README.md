*Part of CEDT02 Embedded Systems Final Project*

## Setup Instructions

### Required Python Packages

- opency-python
- numpy
- gspread
- google-auth

**Installation command**

      pip install opencv-python numpy gspread google-auth

### YOLOv3

- Download the model from [here](https://pjreddie.com/media/files/yolov3.weights) and put it in the same directory as the rest of the app.

### Google Cloud Service Account 

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or use an existing one).
3. Enable Google Sheets API and Google Drive API
4. Go to `Credentials` -> `Create Credentials` -> `Service Account`.
5. Create a service account and download the JSON key.
6. Save the file as `google-credentials.json` in the app directory.
   - Filename can be configured within the source code.

### Google Sheets

1. Create a new Google Sheet named `Person Detection Log`.
   - Google Sheet name can be configured within the code, just like the credentials filename.
2. Share the sheet with your service account email.
   - The email looks like `xxx@xxx.iam.gserviceaccount.com`.
3. Give your service account the `Editor` permission.
