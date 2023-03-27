import numpy as np
from tkinter import filedialog
from PIL import Image
import os
import cv2

# ============================================================================

#BGR
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = np.array(img)
            
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 2)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, 2)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        #User finised entering the polygon points, so let's make the final drawing
        #canvas = np.zeros(CANVAS_SIZE, np.uint8)
        canvas = np.array(img)
        # of a filled polygon
        if (len(self.points) > 0):

            for i in range(CANVAS_SIZE[0]):
                for j in range(CANVAS_SIZE[1]):
                    if cv2.pointPolygonTest(np.array([self.points]), (i,j), False) < 0:
                        canvas[j,i,0] = 0
                        canvas[j,i,1] = 0
                        canvas[j,i,2] = 0
            
            #cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR) 
            #for i in range(1024):
            #    for j in range(1024):
            #        if canvas[i,j,0] != FINAL_LINE_COLOR[0] and canvas[i,j,1] != FINAL_LINE_COLOR[1] and canvas[i,j,2] != FINAL_LINE_COLOR[2]: 
            #            canvas[i,j,0] = 0
            #            canvas[i,j,1] = 0
            #            canvas[i,j,2] = 0
 
            #for i in range(1024):
            #    for j in range(1024):
            #        if canvas[i,j,0] == FINAL_LINE_COLOR[0] and canvas[i,j,1] == FINAL_LINE_COLOR[1] and canvas[i,j,2] == FINAL_LINE_COLOR[2]:
            #            canvas[i,j,0] = img[i,j,0]
            #            canvas[i,j,1] = img[i,j,1]
            #            canvas[i,j,2] = img[i,j,2]

            #cv2.polylines(canvas, np.array([self.points]), True, (255, 255, 255), 3)

        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

# ============================================================================

if __name__ == "__main__":
    path = r"C:\Users\albon\Desktop\Semantic Segmentation 2.0"

    file_types = [('Image', '*.jpg;*.png'), ('All files', '*')]
    name = filedialog.askopenfilename(title='Select a file', filetypes=file_types, initialdir=path)
    img = cv2.imread(name, -1)
    CANVAS_SIZE = [np.array(img).shape[0], np.array(img).shape[1]]
    
    pd = PolygonDrawer("Polygon")
    image = pd.run()
    #cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)