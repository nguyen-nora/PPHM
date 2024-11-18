# import the library 
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import sys
import cvzone 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
vid.set(3,1000)
vid.set(4,480)
vid.set(10,150)

#Yolo Setup
trained = YOLO("best.pt")

# Initialize PaddleOCR
def get_plates(result, img):
    images = [] # Store all license plates
    boxes = result[0].boxes # List of all coordinates of license plates
    img = img.copy()
    for b in boxes:
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1])
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        images.append(img[y1:y2, x1:x2].copy())
    return images

#OCR Image
def format_ocr(plate_text):
    result = ''

    #plate_text = ocr_model.ocr(plate, cls=False, det=True)
    '''
    [[[[[17.0, 5.0], [112.0, 5.0], [112.0, 49.0], [17.0, 49.0]], ('66-B1', 0.9621931910514832)], [[[11.0, 51.0], [132.0, 51.0], [132.0, 98.0], [11.0, 98.0]], ('456.78', 0.9933931231498718)]]]    '''
    
    if len(plate_text[0]) == 2:
        first_line = plate_text[0][0][1][0]
        second_line = plate_text[0][1][1][0]
        print(second_line)
        results = first_line + second_line
        result = result.join(results)
        
    elif len(plate_text[0]) == 1:
        second_line = plate_text[0][0][1][0]
        result = second_line

    #print(result)
        
    if len(result) < 8:
        return None 
    return result
#run OCR
def paddleocr_predict(plate):
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # Use PaddleOCR to extract text from the license plate image
    result = ocr.ocr(plate, cls=True)
    number = result
    return number

#get number after run OCR
def get_LP_number(result, img):
    plates = get_plates(result, img)
    plate_numbers = [] # Store all LP number 
    
    for plate in plates:
        number = paddleocr_predict(plate)
        plate_numbers.append(number)
    
    return plate_numbers

# Process single image
# Draw rectangle around plates and LP number
def draw_box(result, img):
    boxes = result[0].boxes # All coordinates of plates
    plate_numbers = get_LP_number(result, img) # All predicted LP number
    
    # For each LP coordinates and each predicted LP number of that LP
    for b, pnum in zip(boxes, plate_numbers): 
        x1 = int(b.xyxy[0][0])
        y1 = int(b.xyxy[0][1]) - 20 # Small adjust to make it looks better
        x2 = int(b.xyxy[0][2])
        y2 = int(b.xyxy[0][3])
        # Draw rectangle around the LP
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Fill background of the predicted LP number
        cv2.rectangle(img, (x1, y1 + 22), ((x2), (y1)), (0, 255, 0), -1)
        text_xy = (x1 + 2, y1 +18)  # Coordinate of predicted LP number
        # add predicted LP number
        # img, text, position, font, font_scale, color, thickness
        # cv2.putText(img, pnum, text_xy, 0, 0.7, 0, 2) 
        text = format_ocr(pnum)
        cvzone.putTextRect(img, f'{text}', [x1 + 8, y1 - 12], thickness=2, scale=1)
    
    return img


#run video  
while(True):    
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    result = trained(frame)
    frame = draw_box(result, frame)
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 