import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw 

def draw_ellipse(target, params):
    x,y,a,b,alpha = params
    w, h = target.size
    img = Image.new("L", (2*w, 2*h))
    x *= w
    y *= h
    a *= w
    b *= h
    x += w/2
    y += h/2
    canvas = ImageDraw.Draw(img)
    canvas.ellipse((x-a/2,y-b/2,x+a/2,y+b/2), fill ="#FFFFFF", outline ="#FFFFFF")
    img = img.rotate(alpha)
    img = img.crop((w/2, h/2, 3*w/2, 3*h/2))
    target.paste(img, (0,0), img)


if __name__ == '__main__':
    w, h = 256, 256
    struct = Image.new("L", (w, h))
    draw_ellipse(struct, (0.5,0.5,0.3,0.7,45))
    draw_ellipse(struct, (0.2,0.2,0.1,0.9,45))
    struct = struct.resize((256, 16), Image.NEAREST)
    struct = np.asarray(struct)
    
    
    plt.matshow(struct, cmap="Greys", extent=[0,1,0,1])
    plt.show()
    
    
    
