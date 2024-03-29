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
        gid = '1T3BK7WURzuVPWASnZ0x9FJvF7Ge_kSbK'
        filename = "dataset1.zip"
    elif dataset == 2:
        gid = '1Bd41_ALDsKNubIu483-NloMPHlVozZ6r'
        filename = "dataset2.zip"
    elif dataset == 3:
        gid = '1i3Mpdl4IQHutHJ8HRygjEASIgrklEuER'
        filename = "dataset3.zip"
    elif dataset == 4:
        gid = '18oFKVp_5zoMxSsybnyHU6eGG9cPRvC-k'
        filename = "dataset4.zip"
    elif dataset == 5:
        gid = '1eaRbIDDyIK7FpWEByRpOnBUpNLm3w_e7'
        filename = "dataset5.zip"
    elif dataset == 6:
        gid = '1PJNEWRg7AkiiAtAlT7SQepYxviflrkGw'
        filename = "dataset6.zip"
    elif dataset == 7:
        gid = '1ogs9wm0P4b7pDe_HZwqcW5gXlWi7qOfo'
        filename = "dataset7.zip"
    elif dataset == 8:
        gid = '1xbec_UGOyW59mxdfA8fIdCUH05q5XBj-'
        filename = "dataset8.zip"
    elif dataset == 9:
        gid = '1Kn9E0DfPXEYy0xzrkFuOrSiduehNxogh'
        filename = "dataset9.zip"
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