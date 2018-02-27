import vision

def find_ball(image):
    colorLower = (25, 100, 2)
    colorUpper = (35, 255, 255)
    
    #image.blur(100)
    image.in_range(colorLower, colorUpper)
    x, y, radius = image.get_coord()
    image.draw_circle(x, y, radius)
    image.show()
    
    
vision.run(find_ball)
