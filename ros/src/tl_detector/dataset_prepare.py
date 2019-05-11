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
    # Specify the dataset 
    dataset = int(sys.argv[1])
    
    # Get id and filename
    if dataset == 1:
        gid = '1NJ8mDGhX16NksFNfpKvPlEt4UVIqhIL3'
        filename = "dataset.zip"
    if dataset == 2:
        gid = '1FfpetULTpRBx62mr23IZ7dCdj7lQw_7o'
        filename = "dataset2.zip"
    else:
        print("Not available yet. To be added")

    # Call function to download
    print("Download the dataset#" + str(dataset))
    file_download(gid, filename)
    print("--------- Download completed -----------")
    
    # Unzip the dataset
    print("Unzip " + filename)
    subprocess.call("unzip " + filename, shell=True)

    # Delete the zip file
    print("Delete zip file " + filename)
    subprocess.call("rm " + filename, shell=True)

if __name__ == "__main__":
    main()