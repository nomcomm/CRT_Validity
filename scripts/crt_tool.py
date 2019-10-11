

import sys, PIL, os
from os.path import join
import numpy as np

from pliers.tests.utils import get_test_data_path
from pliers.extractors import (FaceRecognitionFaceLocationsExtractor, FaceRecognitionFaceEncodingsExtractor, 
                               FaceRecognitionFaceLandmarksExtractor,SaliencyExtractor, BrightnessExtractor, merge_results)
from pliers.filters import FrameSamplingFilter
import face_recognition
from pliers.graph import Graph
import pandas as pd 


import matplotlib.pyplot as plt
import matplotlib.patches as patches

get_ipython().run_line_magic('matplotlib', 'inline')


def crt_process(    video_reference = '../data/input_data/videos/Shark_Tank/Shark_Tank.mp4',
                    reference_image_dir = '../data/input_data/reference_images/Shark_Tank/',
                    ):
    
    #Name the Project Below
    project_name = 'My_First_CRT_Analysis'
    project_name
    
    #what's this needed for?

    
    #Import Video
    print('This project will use ' + str(video_reference) + ' as the input video.')
    

    #Import Reference Images 
    character_images = [i for i in os.listdir(reference_image_dir) if i.endswith('.jpg')]
    n_character_images = len(character_images) 
    print('The CRT will search for ' + str(n_character_images) + ' characters.')
    
    #What are the Names of the Characters (Alphabetical)?
    character_names = [x[:-4] for x in character_images ]
    n_characters = len(character_names)
    print('The CRT will identify the following characters: ')
    
    plt.figure(figsize=(18,4))
    for num, x in enumerate(character_images):
        img = PIL.Image.open(os.path.join(reference_image_dir, x))
        plt.subplot(1,n_characters ,num+1)
        plt.title(x.split('.')[0])
        plt.axis('off')
        plt.imshow(img)
    
   
    # change to value you prefer (values near 1 are more accurate, higher values will skip frames...)
    rate = 4450 
    print('We will work with a frame rate of ' + str(rate) + ' frames.')
    
    # change to values between 0 and 1 - values towards 0 are more strict, higher values more liberal,
    # usually values around 0.6 give best performance
    tolerance_threshold = 0.6
    
    model_to_use ='hog' #'hog' is faster, 'cnn' slower, but more accurate (if cnn, a gpu is advised)
    
    sampler = FrameSamplingFilter(every = rate)
    frames = sampler.transform(video_reference)
    n_frames = frames.n_frames
    n_frames_analyze = round(n_frames/24)
    print('The current extraction will work on ' + str(n_frames) + ' frames.')
    if (n_frames) < 100:
        print('This should be pretty quick')
    else: 
        print('This may take a while')
    
    
    known_faces = []
    
    for character in character_images:
        known_character_image =  face_recognition.load_image_file(reference_image_dir + character)
        known_character_encoding = face_recognition.face_encodings(known_character_image)[0]
        #print (character)
        known_faces.append(known_character_encoding)
        
    plt.figure(figsize = (14,8))
    plt.imshow(known_faces, cmap = 'seismic');
    plt.title('Face encoding vectors for the to-be-recognized characters')
    #plt.ylabel(character_images)
    #plt.colorbar();
    print(str(len(known_faces)) + ' character templates have been encoded and can now be searched.')
    

    
    # ## Go through frames, detect faces, match them to templates, and create output
    # 
    # This is the meat of the code....

    
    unkonown_face_counter = 0
    res = []
    for curr_frame_number in np.arange(0, n_frames, 1):
        sys.stdout.write(" %d, \r" % (curr_frame_number) )
        sys.stdout.flush()
        
        # load the current frame as an image
        curr_frame = frames.get_frame(curr_frame_number).video.get_frame(curr_frame_number).video.get_frame(curr_frame_number).data
    
        # display for now
        fig, ax = plt.subplots(1)
        ax.imshow(curr_frame)
    
        # detect faces and plot them (for now) based on location info...
        face_locations = face_recognition.face_locations(curr_frame, model= model_to_use) #'number_of_times_to_upsample=1', "model='hog'"
        if len(face_locations)>0:
            #print('I see a face!')
            for curr_face in range(len(face_locations)):
            
                l1 = (face_locations[curr_face][2] - face_locations[curr_face][0])
                l2 = (face_locations[curr_face][1] - face_locations[curr_face][3])
                rect = patches.Rectangle((face_locations[curr_face][3], face_locations[curr_face][0]), l1,l2, edgecolor = 'r', facecolor='none')
    
                ax.add_patch(rect)
                
    
        plt.axis('off')
        plt.show()
        sys.stdout.flush()
        
        
        # for all detected faces, compute their encodings....
        face_encodings = face_recognition.face_encodings(curr_frame, face_locations)
    
        if len(face_encodings)>0 :
            results = []
            #n_recognized = 0
            for curr_encoding in range(len(face_encodings)):
                # compare the recognized & encoded faces with the known faces' encodings....
                results.append(face_recognition.compare_faces( known_faces, 
                                                          face_encodings[curr_encoding], 
                                                          tolerance = tolerance_threshold))
            results = np.sum(results, axis=0, dtype = 'bool') 
            #n_recognized = 
            
            if (sum(results)>0):
                
                characters_recognized = [ character_names[i] for i in np.where(results)[0][:] ]
                #print(characters_recognized)
                print('I see the face of ' + str(characters_recognized) )
                
                #print('I see the face of ' + character_names[np.where(results)[0][0]])
                #print(results)
                #print()
                print('--')
                
            #### This is where we do the surgery to add the unknown/feedback/human in the loop part
            
            # if more faces are detected than recognized, then we conclude that new faces are present
            if (len(face_encodings) > sum(results)):
                print('I see a face that I do not recognize, I save it to the reference_images folder.' )
                
                #cut out the new faces and put them in folder...
                #for current_face_seen in range( len(face_encodings[0]) ):
                
                print(face_locations[0])
                cc = curr_frame[    face_locations[0][0]:face_locations[0][2], 
                                    face_locations[0][3]:face_locations[0][1],
                                    :]
    
                plt.imshow(cc)
                plt.axis('off')
                plt.show()
                
                from PIL import Image
                im = Image.fromarray(cc)
                unkonown_face_counter += 1
                filename_uf = '../data/input_data/unknown_reference_images/unknown_face_' + str(unkonown_face_counter) + '.jpg'
                
    
                im.save(filename_uf)
                
            
            
            #append the results...
            res.append(results)
        else:
            res.append([False] * n_character_images)
            
    res2 = np.asarray(res)
    np.asarray(res2[0])
    
    new_result = np.zeros((n_frames,n_characters))
    
    for this_line in range(n_frames):
        new_result[this_line,:] = np.asarray(res2[this_line])
    
    df2 = pd.DataFrame( data    = new_result,
                        columns = character_names );
    df2.to_csv('../data/output_data/' + project_name + '_face_recognition.csv')
    df2
    
    return df2
    
    
    

