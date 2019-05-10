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
        # A small subset of sample data (30 images) for quick debug
        gid = '1Vik42Q_qMV_0tlXBZK9WxvYM3tToxQU2'
        filename = "dataset1.zip"
    elif dataset == 2:
        # Sample data (from lecture)
        gid = '1CWP4K-HlOqaOUXTMAQ8_vPP5FzP1pYwH'
        filename = "dataset2.zip"
    elif dataset == 3:
        # Simulated data (from simulator)
        # 2.5 round counter clockwise
        gid = '1cndsQ6f-0ndbBEWUgZdX0sSomv1PZPap'
        filename = "dataset3.zip"
    elif dataset == 4:
        # Simulated data (from simulator)
        # 3 round clockwise
        gid = '1akOvfxOe8Ln8BhxBDGzFX2XxMhqLtdia'
        filename = "dataset4.zip"
    elif dataset == 5:
        # Simulated data (from simulator)
        # Additional data to improve the behavior on quick & difficult curve
        gid = '1Agmy5leXNWjQTVCnmfsGnOWrZtr-UxWh'
        filename = "dataset5.zip"
    elif dataset == 6:
        # Simulated data (from simulator)
        # Additional data to improve the drive on specific place (keep center)
        gid = '1ir9Gyi0pglAeYynQIBt138mgNIsKcukn'
        filename = "dataset6.zip"
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