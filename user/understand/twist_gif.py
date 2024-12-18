'''
Compute the fields for different twist angles and make a gif.
'''
cl, pic = build_crystal(np.rad2deg(params[4]))
fig, ax = plt.subplots()

xres, yres = 512, 128
zres = 128
xcells = 12
extent = [0, xcells, 0, 4+3]
im = ax.matshow(pic, extent=extent, cmap="RdBu", origin="lower", vmin=-1, vmax=1)
ax.set(xlim=[0, xcells], ylim=[0, 7], xlabel='X [um]', ylabel='Y [um]')
ax.legend()
ax.axis("equal")
N = 128
angles = np.linspace(0.01, 59.99, N)

def update(frame):
cl, _ = build_crystal(np.rad2deg(params[4]), ta=angles[frame])
cl.set_source(1.01, te=1,tm=0)
cl.solve()

x, y, z = coords(0, xcells, 0.0, 1.0, -0.1, cl.depth+3, (xres, yres, zres))

E, H = cl.fields_volume(x, y, z)
# for each frame, update the data stored on each artist.
im.set_data(E[:, 0, :, yres//2].real)
ax.set_title(str(frame))
return (im)


ani = animation.FuncAnimation(fig=fig, func=update, frames=N, interval=200)
ani.save("figs/tilt.gif")