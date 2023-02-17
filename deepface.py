from deepface import DeepFace
import pandas as pd
import cv2

camera_capture = cv2.VideoCapture(0)

fps = 30
size = int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),\
        int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#vidwrite = cv2.VideoWriter(r'C:\Users\RX971WG\OneDrive - EY\Documents\GitHub\FER\Input\FER_Videos\testvideo3.mp4', cv2.VideoWriter_fourcc('I','4','2','0'), fps, 
           #size,False
flag=True
# Write and show recording
counter=0
while (flag):

        _, frame =  camera_capture.read()
        #vidwrite.write(frame)
        
 
        
        try:    
                

               
                k=DeepFace.analyze( frame,actions = ['emotion'])
                x1=k['region']['x']
                y1=k['region']['y']
                x2=k['region']['x']+k['region']['w']
                y2=k['region']['y']+k['region']['h']
                frame=cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                frame=cv2.putText(frame, k['dominant_emotion'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                if counter==0:
                        data=pd.DataFrame([k['emotion'].values()],columns=k['emotion'].keys())
                else:
                        data=pd.concat([data,pd.DataFrame([k['emotion'].values()],columns=k['emotion'].keys())],ignore_index=True)
        
                #else:
                        #frame=cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                        #frame=cv2.putText(frame, k['dominant_emotion'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        except Exception as e :
                print(e)
                pass
        counter +=1
        cv2.imshow("showlive", frame)
        
        if cv2.waitKey(30) & 0xff == ord('q'):
            flag=False
            #vidwrite.release()
            cv2.destroyAllWindows()
            break
