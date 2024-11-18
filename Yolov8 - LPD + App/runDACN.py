from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import sys
import cvzone


# Get all license plates in image
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

#from paddleocr import PaddleOCR
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

# Process video 
def video_draw_box(vid_path, model):
    cap = cv2.VideoCapture(vid_path)
    
    # width and height must be correct to save output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20, (width, height))
    
    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
       
        result = model(frame)   # Predict position of LPs
        frame = draw_box(result, frame) # Draw rectangle and predicted LP number for current frame
        
        out.write(frame)    # Write to output.avi
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    # cap.release()
    out.release()
    cv2.destroyAllWindows()

if len(sys.argv) == 3:
    # Get weights and img_dir
    pre_trained_model = "best.pt"
    
    media_type = sys.argv[1]
    
    file_dir = sys.argv[2]
    
    # Create model
    model = YOLO(pre_trained_model)
   
    if media_type == "-image":
        img = cv2.imread(file_dir)
        result = model(img)
        img = draw_box(result, img)
        cv2.imshow("Result", img)
        cv2.waitKey(0)
        cv2.imwrite("predicted-" + file_dir, img)
        
    elif media_type == "-video":
        video_draw_box(file_dir, model)