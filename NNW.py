import sys
import os
import argparse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import csv

def summarized_output(image_list, outputfile):
    # File to be written to containing csv data of images
    heading = ['imagename', 'latitude', 'longitude', 'altitude']
    
    # catch open failing (move to main?)
    f = open(outputfile, 'w')

    writer = csv.writer(f)
    
    # Image processing begins here
    for each in os.listdir(image_list):
        filename = os.path.basename(each)
        filename = os.path.join(image_list, filename)
        new_image = Image.open(filename)
        exifdata = new_image._getexif()

        if exifdata is not None:
            for key, value in exifdata.items():
                Name = TAGS.get(key, key)
                exifdata[Name] = exifdata.pop(key)

        if 'GPSInfo' in exifdata:
            for key in exifdata['GPSInfo'].keys():
                Name = GPSTAGS.get(key, key)
                exifdata['GPSInfo'][Name] = exifdata['GPSInfo'].pop(key)
        else:
            print("{} Image has no GPS exif data and will be skipped".format(filename))
            continue

        #Calculation to decimal for Latitude
        if 'GPSLatitude' in exifdata['GPSInfo'] and 'GPSLatitudeRef' in exifdata['GPSInfo']:
            e = exifdata['GPSInfo']['GPSLatitude']
            ref = exifdata['GPSInfo']['GPSLatitudeRef']
            degrees = e[0]
            minutes = e[1] / 60.0
            seconds = e[2] / 3600.0
            Latitude = round(degrees + minutes + seconds, 6)
            
        #Checking for neccessity of negative if South
        if exifdata['GPSInfo']['GPSLatitudeRef'] == 'S':
            Latitude = Latitude * -1


        #Calculation to decimal for Longitude
        if 'GPSLongitude' in exifdata['GPSInfo'] and 'GPSLongitudeRef' in exifdata['GPSInfo']:
            e = exifdata['GPSInfo']['GPSLongitude']
            ref = exifdata['GPSInfo']['GPSLongitudeRef']
            degrees = e[0]
            minutes = e[1] / 60.0
            seconds = e[2] / 3600.0
            Longitude = round(degrees + minutes + seconds, 6)
            
        #Checking for neccessity of negative if West    
        if exifdata['GPSInfo']['GPSLongitudeRef'] == 'W':
            Longitude = Longitude * -1

        
        #Adding Altitude to the mix of data
        if 'GPSAltitude' in exifdata['GPSInfo']:
            Altitude = exifdata['GPSInfo']['GPSAltitude']
        else:
            Altitude = 'N/A'

        print(Latitude, Longitude, Altitude)
        
        
        #Writing new line of data to file
        data = [filename, Latitude, Longitude, Altitude]
        writer.writerow(data)
        
    f.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Provides CSV output of longitude, latitude and altitude of image')
    parser.add_argument('-i', '--inputfiles', nargs='+', type=str, help='Path to images')
    parser.add_argument('-o', '--outputfilename', type=str, help='CSV file name to store output')
    args = parser.parse_args()
    if args.inputfiles is None:
        print("Error, no input files were provided.")
        exit(-1)
    
    if args.outputfilename is None:
        print("Error, no output file was provided.")
        exit(-1)
        
    directory = args.inputfiles[0]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if not(os.path.isfile(f)):
            print("Error, summarize-images.py. File", everyfile, "does not exist, is not readable, etc.", file=sys.stderr)
            exit(-1)

    summarized_output(directory, args.outputfilename)

    exit(0)


