import vision

lol = []

def find_ball(image):
    colorLower = (15, 120, 2)
    colorUpper = (25, 255, 255)
    
    image.blur(30)
    image.in_range(colorLower, colorUpper)
    image.erode(10)
    image.dilate(10)
    
    x, y, radius = image.min_enclosing_circle(40)
    image.draw_circle(x, y, radius)

    if len(lol) > 10:
        lol.remove(lol[0])
        
    lol.append((int(x), int(y)))

    image.draw_trail(lol)

    image.show()
    
    
vision.run(find_ball)
