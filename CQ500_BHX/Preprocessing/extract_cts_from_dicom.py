import os
import pandas as pd
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_windowing
from preprocessing import remove_noise, remove_tilt, crop_image, add_pad
import cv2
import numpy as np
from natsort import natsort_keygen
import matplotlib.pyplot as plt


# patient 120 is faulty. Cannot unzip!
root = "/Volumes/Masterarbeit/"

# Train- Validation- or Test Data
datafolder = "Train/"
ids_for_bb = pd.DataFrame([], columns=['label','uniqueID','M','topleft','bottomright'])

# root path to data
cts_unzipped = root + "CTdata/unzipped_qureai/" + datafolder

# csv file with all boudning boxes (hand drawn and extrapolated)
labeled = root + "CTdata/BHX/physionet.org/files/bhx-brain-bounding-box/1.1/3_Extrapolation_to_Selected_Series.csv"
df = pd.read_csv(labeled)

# get list of unique identifiers of cts (combination of SOPInstanceUID, SeriesInstanceUID, and StudyInstanceUID)
ids = []
labels = []
handDrawn = []
for index, row in df.iterrows():
    # get unique id
    unique_id = row["SOPInstanceUID"] + row["SeriesInstanceUID"] + row["StudyInstanceUID"]
    # append unique_id to list of ids
    ids.append(unique_id)
    # get label
    labels.append(row["labelName"])
    handDrawn.append(row["labelType"])

# sanity check: should be 27203 bounding boxes
print(len(ids))
# sanity check: should be 15979 cts (multiple bounding boxes in one ct possible)
print(len(set(ids)))
# sanity check: set of labels {'Chronic', 'Epidural', 'Subarachnoid', 'Intraparenchymal', 'Subdural', 'Intraventricular'}
print(set(labels))

###########################################################################
#### Divide the CTs by intraventricular and non intraventricular class ####
####                     For binary classification                     ####
###########################################################################

# check if folder exists, if not create it
if not os.path.exists(root + "Classification_rotated/" + datafolder + "Hemorrage"):
    os.makedirs(root + "Classification_rotated/" + datafolder + "Hemorrage")

# check if folder exists, if not create it
if not os.path.exists(root + "Classification_rotated/" + datafolder + "Not_Hemorrage"):
    os.makedirs(root + "Classification_rotated/" + datafolder + "Not_Hemorrage")

# path to classification data
class_path = root + "Classification_rotated/"

# get list of all patients
patients = os.listdir(cts_unzipped)  # 491 patients

hem_counter = 0
not_hem_counter = 0
# iterate over patients
for pat in patients:
    # there should only be one subfolder
    if len(os.listdir(cts_unzipped + pat)) == 1:
        pat_1 = cts_unzipped + pat + "/" + os.listdir(cts_unzipped+pat)[0]
        # there should only be one subsubfolder
        if len(os.listdir(pat_1)) == 1:
            pat_2 = pat_1 + "/" + os.listdir(pat_1)[0]
            # go into every CT folder and extract SOP Instance, Study, Series UID
            for folder in os.listdir(pat_2):
                cts_path = pat_2 + "/" + folder
                # get the SOP Instance, Study, Series UID for every file
                for file in os.listdir(cts_path):
                    fpath = cts_path + "/" + file
                    ds = dcmread(fpath)
                    id = ds.SOPInstanceUID + ds.SeriesInstanceUID + ds.StudyInstanceUID
                    # if the id is in the list of IDs => hemorrage in the image
                    # save the dcm file to 'hemorrage' folder
                    if id in ids:
                         # update counter
                        hem_counter = hem_counter + 1
                        # preprocess
                        img = remove_noise(fpath, display=False)
                        try: # tilt correction throws error in some cases (i.e. empty images): these images are not used for further analysis
                            img, M = remove_tilt(img)    
                            img, top_left, bottom_right = crop_image(img)
                            img = add_pad(img)                      
                        except:
                            continue
                        # copy dcm file into folder
                        # save as jpg
                        cv2.imwrite(class_path + datafolder + "Hemorrage/Hemorrage" + str(hem_counter) + ".jpg", img)

                        image_label = "Hemorrage" + str(hem_counter)
                        image_id = id
                        image = pd.DataFrame([[image_label, image_id, [M], [top_left], [bottom_right]]], columns = ['label','uniqueID','M','topleft','bottomright'])
                        ids_for_bb = ids_for_bb.append(image)
                    
                    else:
                        # update counter
                        not_hem_counter = not_hem_counter + 1
                        # preprocess
                        img = remove_noise(fpath, display=False)
                        try: # tilt correction throws error in some cases (i.e. empty images): these images are not used for further analysis
                            img, M = remove_tilt(img) 
                            img, top_left, bottom_right = crop_image(img)
                            img = add_pad(img)                         
                        except:
                            continue
                        # copy dcm file into folder
                        # save as jpg
                        cv2.imwrite(class_path + datafolder + "Not_Hemorrage/Not_Hemorrage" + str(not_hem_counter) + ".jpg", img)

                        image_label = "Not_Hemorrage" + str(not_hem_counter)
                        image_id = id
                        image = pd.DataFrame([[image_label, image_id, [M], [top_left], [bottom_right]]], columns = ['label','uniqueID','M','topleft','bottomright'])
                        ids_for_bb = ids_for_bb.append(image)



# Save dictionaries
ids_for_bb['helper'] = ids_for_bb['label']
ids_for_bb['helper'] = ids_for_bb['helper'].str.replace('Not','A_NOT').copy()
ids_for_bb_sorted = ids_for_bb.sort_values(by = 'helper',key = natsort_keygen() )
ids_for_bb_sorted.drop('helper',axis = 1)

ids_path = root + "Classification_rotated/" + datafolder
ids_for_bb_sorted.to_csv(ids_path+ "ids_for_bb.csv")