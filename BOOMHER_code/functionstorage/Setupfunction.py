def setup(image_path,loopnumber):
    #!git clone https://github.com/ultralytics/yolov5  # clone
    %cd yolov5
    
    if loopnumber == 0:
        
        import utils
        
        #!pip install -qr requirements.txt comet_ml  # install
        display = utils.notebook_init()  # checks
        random.seed(108)
        
        

    image = Image.open(image_path)
    

        # clean up directory to make space for new slices of the image 
    
    # Specify the directory path
    save_directory = here + "\\cropdump\\"  # Replace with the actual folder path
    
    # if save_directory not exists make it 
    
    
    # List all files in the directory
    file_list = os.listdir(save_directory)

    # Iterate through the files and delete them
    for filename in file_list:
        file_path = os.path.join(save_directory, filename)

        # Check if the path is a file (not a directory) and delete it
        if os.path.isfile(file_path):
            os.remove(file_path)

    print(f"All files in {save_directory} have been deleted.")

    #try:
    # Specify the directory path
    runs = here + "\\yolov5\\runs\\detect\\"  # Replace with the actual folder path
    

    # List all files and directories in the directory
    entries = os.listdir(runs)

    # Iterate through the entries and delete them
    for entry in entries:
        entry_path = os.path.join(runs, entry)

        if os.path.isfile(entry_path):
            # Delete files
            os.remove(entry_path)
        elif os.path.isdir(entry_path):
            # Delete directories and their contents
            shutil.rmtree(entry_path)  # Use shutil.rmtree() to delete non-empty directories

    print(f"All files and directories in {runs} have been deleted.")
    #except:
    #    print("failed to delete stuff")
    return image