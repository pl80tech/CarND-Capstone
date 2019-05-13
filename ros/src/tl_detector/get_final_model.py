import sys
import subprocess

def file_download(gid, filename):
    # Install GoogleDriveDownloader to download file from Google Drive
    subprocess.call("pip install googledrivedownloader", shell=True)
    
    # Import necessary package
    from google_drive_downloader import GoogleDriveDownloader as gdd
    
    # Download file
    savepath = "./" + filename
    gdd.download_file_from_google_drive(file_id=gid, dest_path=savepath, unzip=False)

# Main function
def main():       
    # Specify the frozen graph of final model 
    model = int(sys.argv[1])
    
    # Get id and filename
    if model == 1:
        gid = '1An4y70YiEyRCFkrsig75iBggEtHTZZHq'
        filename = "frozen_inference_graph.pb"
    else:
        print("Not available yet. To be added")

    # Call function to download
    print("Download the model#" + str(model))
    file_download(gid, filename)
    print("--------- Download completed -----------")
    
    # Copy the downloaded graph to specified path
    print("Move the downloaded graph to specified path " + filename)
    subprocess.call("mv " + 'model/final/', shell=True)

if __name__ == "__main__":
    main()