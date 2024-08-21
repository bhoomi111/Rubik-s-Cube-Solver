import cv2
import numpy as np
import time

def detect_color_in_cell(cell):
    hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for Rubik's Cube colors
    color_ranges = {
        'white': ((0, 0, 200), (180, 60, 255)),
        'yellow': ((20, 100, 100), (30, 255, 255)),
        'red': ((0, 100, 100), (10, 255, 255)),
        'orange': ((10, 100, 100), (20, 255, 255)),
        'blue': ((90, 100, 100), (130, 255, 255)),
        'green': ((35, 100, 100), (85, 255, 255))
    }
    
    detected_colors = {}
    
    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the current color
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Count the number of pixels for the color
        color_pixels = cv2.countNonZero(mask)
        
        # Store the color and its pixel count
        detected_colors[color] = color_pixels
    
    # Find the color with the maximum pixel count
    max_color = max(detected_colors, key=detected_colors.get)
    return max_color

def draw_grid(frame):
    height, width, _ = frame.shape
    grid_size = min(height, width) // 3
    
    # Calculate starting points to center the grid
    start_x = (width - grid_size * 3) // 2
    start_y = (height - grid_size * 3) // 2
    
    # Draw horizontal lines
    for i in range(4):
        y = start_y + i * grid_size
        cv2.line(frame, (start_x, y), (start_x + grid_size * 3, y), (0, 255, 0), 2)

    # Draw vertical lines
    for i in range(4):
        x = start_x + i * grid_size
        cv2.line(frame, (x, start_y), (x, start_y + grid_size * 3), (0, 255, 0), 2)
    
    return start_x, start_y, grid_size
url = 'http://192.0.0.4:8080/video'

# Create a VideoCapture object
cap = cv2.VideoCapture(url)

# Open a connection to the camera
last_output_time = time.time()
output_interval = 10  # Output interval in seconds

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break
    
    # Draw the centered 3x3 grid on the frame
    start_x, start_y, grid_size = draw_grid(frame)

    # Create a 3x3 matrix to store the detected colors
    color_matrix = [[''] * 3 for _ in range(3)]

    # Process each grid cell
    for i in range(3):
        for j in range(3):
            x1 = start_x + j * grid_size
            y1 = start_y + i * grid_size
            x2 = x1 + grid_size
            y2 = y1 + grid_size
            
            # Extract the cell from the frame
            cell = frame[y1:y2, x1:x2]
            
            # Detect the predominant color in the cell
            color = detect_color_in_cell(cell)
            color_matrix[i][j] = color
            
            # Optionally, draw the detected color in the frame
            cv2.putText(frame, color, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame with the grid and color labels
    cv2.imshow('Camera Feed with Grid and Colors', frame)

    # Check if it's time to output the color matrix
    current_time = time.time()
    if current_time - last_output_time >= output_interval:
        print("Color Matrix:")
        for row in color_matrix:
            print(row)
        last_output_time = current_time

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
