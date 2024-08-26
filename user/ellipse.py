import numpy as np

from PIL import Image, ImageDraw 

def draw_ellipse(canvas, params):
    X, Y = np.meshgrid(np.linspace(-0.5, 0.5, canvas.shape[1]), 
                       np.linspace(-0.5, 0.5, canvas.shape[0]))
    x, y, a, b, alpha = params
    A = (np.cos(alpha)**2 / a**2) + (np.sin(alpha)**2 / b**2)
    B = (np.cos(alpha)**2 / b**2) + (np.sin(alpha)**2 / a**2)
    C = 2 * np.cos(alpha) * np.sin(alpha) * (1/a**2-1/b**2)
    dsqr = (X-x)**2 + (Y-y)**2
    theta = np.arctan2(X-x, Y-y)
    irsqr = A*np.cos(theta)**2+B*np.sin(theta)**2+C*np.sin(theta)*np.cos(theta)
    canvas[dsqr<1.0/irsqr] = 1.0




def draw_ellipse_old(target, params):
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
    return target
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    w, h = 256, 16
    struct = np.zeros((h,w))#, dtype=np.uint8)
    x = np.linspace(-0.5, 0.5, w)
    y = np.linspace(-0.5, 0.5, h)
    X, Y = np.meshgrid(x, y)
    draw_ellipse(struct, (0,0, 0.25,0.1, np.pi/4))
    fig, ax = plt.subplots(figsize=(6,6)) 
    ax.matshow(struct, cmap="Greys", extent=[0,1,0,1])
    plt.savefig("el.png")    
    
    
